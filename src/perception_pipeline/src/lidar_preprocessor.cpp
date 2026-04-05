#include "perception_pipeline/lidar_preprocessor.hpp"

#include <numeric>
#include <vector>

#include <open3d/geometry/BoundingVolume.h>

namespace perception_pipeline {

LidarPreprocessor::LidarPreprocessor(Config cfg)
: cfg_(std::move(cfg)) {}

PreprocessResult LidarPreprocessor::process(
    const std::vector<Eigen::Vector3d>& points) const
{
  PreprocessResult result;
  result.stats.n_input = points.size();

  // ── Stage 1: ROI crop ─────────────────────────────────────────────────────
  // Python: boolean mask on xyz within [min, max] per axis
  // C++:    AxisAlignedBoundingBox::Crop — same O(N) semantics
  open3d::geometry::PointCloud raw_pcd;
  raw_pcd.points_ = points;

  const open3d::geometry::AxisAlignedBoundingBox roi_box(
    Eigen::Vector3d(cfg_.roi_x_min, cfg_.roi_y_min, cfg_.roi_z_min),
    Eigen::Vector3d(cfg_.roi_x_max, cfg_.roi_y_max, cfg_.roi_z_max)
  );
  auto roi_pcd = raw_pcd.Crop(roi_box);
  result.stats.n_roi = roi_pcd->points_.size();

  if (result.stats.n_roi < 10) {
    return result;
  }

  // ── Stage 2: Voxel downsampling ───────────────────────────────────────────
  // Python: o3d.geometry.PointCloud.voxel_down_sample(voxel_size)
  // C++:    identical API, same parameter name
  auto voxel_pcd = roi_pcd->VoxelDownSample(cfg_.voxel_size);
  result.stats.n_voxel = voxel_pcd->points_.size();

  if (result.stats.n_voxel < 10) {
    return result;
  }

  // ── Stage 3: Distance filter ──────────────────────────────────────────────
  // Python: dist = np.linalg.norm(xyz, axis=1); mask = dist <= max_depth
  // C++:    collect indices where norm <= max_depth, then SelectByIndex
  std::vector<size_t> close_indices;
  close_indices.reserve(voxel_pcd->points_.size());
  for (size_t i = 0; i < voxel_pcd->points_.size(); ++i) {
    if (voxel_pcd->points_[i].norm() <= cfg_.max_depth) {
      close_indices.push_back(i);
    }
  }
  auto dist_pcd = voxel_pcd->SelectByIndex(close_indices);

  if (dist_pcd->points_.size() < 10) {
    return result;
  }

  // ── Stage 4: RANSAC ground plane removal ──────────────────────────────────
  // Python: pcd.segment_plane(distance_threshold, ransac_n=3, num_iterations)
  //         returns (plane_model, inlier_indices)
  // C++:    identical signature — returns std::tuple<Eigen::Vector4d, vector<size_t>>
  auto [plane_model, inlier_indices] = dist_pcd->SegmentPlane(
    cfg_.ransac_dist,
    /*ransac_n=*/3,
    cfg_.ransac_iter
  );
  (void)plane_model;  // plane equation not used downstream

  // Python: non_ground = set complement of inlier_indices
  // C++:    SelectByIndex with invert=false (ground) and invert=true (non-ground)
  auto ground_pcd   = dist_pcd->SelectByIndex(inlier_indices, /*invert=*/false);
  auto filtered_pcd = dist_pcd->SelectByIndex(inlier_indices, /*invert=*/true);

  result.stats.n_ground = ground_pcd->points_.size();
  result.stats.n_output = filtered_pcd->points_.size();

  result.filtered = std::move(*filtered_pcd);
  result.ground   = std::move(*ground_pcd);

  return result;
}

}  // namespace perception_pipeline
