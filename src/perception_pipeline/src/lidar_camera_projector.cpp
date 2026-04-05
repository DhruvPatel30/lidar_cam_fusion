#include "perception_pipeline/lidar_camera_projector.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace perception_pipeline {

LidarCameraProjector::LidarCameraProjector(const ProjectorConfig & cfg)
: cfg_(cfg) {}

// ── project_to_image ─────────────────────────────────────────────────────────

std::pair<std::vector<Eigen::Vector2d>, std::vector<std::size_t>>
LidarCameraProjector::project_to_image(
  const std::vector<Eigen::Vector3d> & points) const
{
  std::vector<Eigen::Vector2d> pixels;
  std::vector<std::size_t>     indices;
  pixels.reserve(points.size());
  indices.reserve(points.size());

  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto & p = points[i];

    // Velodyne homogeneous point → rectified camera frame
    const Eigen::Vector4d p_velo(p.x(), p.y(), p.z(), 1.0);
    const Eigen::Vector4d p_rect = cfg_.T * p_velo;

    if (p_rect.z() <= 0.0) {
      continue;  // behind the camera
    }

    // Project onto image plane
    const Eigen::Vector3d p_img = cfg_.P * p_rect;
    const double u = p_img.x() / p_img.z();
    const double v = p_img.y() / p_img.z();

    if (u >= 0.0 && u < static_cast<double>(cfg_.img_width) &&
        v >= 0.0 && v < static_cast<double>(cfg_.img_height))
    {
      pixels.emplace_back(u, v);
      indices.push_back(i);
    }
  }

  return {std::move(pixels), std::move(indices)};
}

// ── fuse ─────────────────────────────────────────────────────────────────────

std::vector<FusedDetection> LidarCameraProjector::fuse(
  const std::vector<Eigen::Vector3d> &        points,
  const std::vector<Eigen::Vector2d> &        pixels,
  const std::vector<std::size_t> &            valid_indices,
  const std::vector<std::array<double, 4>> &  bboxes,
  const std::vector<std::string> &            class_ids,
  const std::vector<double> &                 scores) const
{
  std::vector<FusedDetection> results;
  results.reserve(bboxes.size());

  for (std::size_t b = 0; b < bboxes.size(); ++b) {
    const double cx = bboxes[b][0];
    const double cy = bboxes[b][1];
    const double hw = bboxes[b][2] * 0.5;  // half-width
    const double hh = bboxes[b][3] * 0.5;  // half-height

    const double x_min = cx - hw;
    const double x_max = cx + hw;
    const double y_min = cy - hh;
    const double y_max = cy + hh;

    // Collect 3-D Velodyne points whose projection falls inside this bbox
    std::vector<Eigen::Vector3d> matched;
    for (std::size_t i = 0; i < pixels.size(); ++i) {
      const double u = pixels[i].x();
      const double v = pixels[i].y();
      if (u >= x_min && u <= x_max && v >= y_min && v <= y_max) {
        matched.push_back(points[valid_indices[i]]);
      }
    }

    if (static_cast<int>(matched.size()) < cfg_.min_cluster_points) {
      continue;
    }

    // Centroid + axis-aligned bounding box size
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    Eigen::Vector3d pmin(
       std::numeric_limits<double>::max(),
       std::numeric_limits<double>::max(),
       std::numeric_limits<double>::max());
    Eigen::Vector3d pmax(
      -std::numeric_limits<double>::max(),
      -std::numeric_limits<double>::max(),
      -std::numeric_limits<double>::max());

    for (const auto & pt : matched) {
      centroid += pt;
      pmin = pmin.cwiseMin(pt);
      pmax = pmax.cwiseMax(pt);
    }
    centroid /= static_cast<double>(matched.size());

    FusedDetection det;
    det.centroid = centroid;
    det.size     = pmax - pmin;
    det.class_id = class_ids[b];
    det.score    = scores[b];
    results.push_back(std::move(det));
  }

  return results;
}

}  // namespace perception_pipeline
