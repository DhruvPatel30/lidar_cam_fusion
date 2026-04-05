#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <open3d/geometry/PointCloud.h>

namespace perception_pipeline {

/// Configuration for LidarPreprocessor — declared at namespace scope so it can
/// be used as a default argument in the constructor without triggering the
/// Clang "default member initializer needed outside member functions" error.
struct LidarPreprocessorConfig {
  double roi_x_min   =  0.0;
  double roi_x_max   = 50.0;
  double roi_y_min   = -10.0;
  double roi_y_max   =  10.0;
  double roi_z_min   = -3.0;
  double roi_z_max   =  2.0;
  double voxel_size  =  0.1;
  double ransac_dist =  0.2;
  int    ransac_iter = 100;
  double max_depth   = 50.0;
};

/// Per-frame processing statistics — mirrors Python's stats dict.
struct PreprocessStats {
  std::size_t n_input  = 0;
  std::size_t n_roi    = 0;
  std::size_t n_voxel  = 0;
  std::size_t n_ground = 0;
  std::size_t n_output = 0;
};

/// Return value of LidarPreprocessor::process().
struct PreprocessResult {
  open3d::geometry::PointCloud filtered;  // non-ground points after all stages
  open3d::geometry::PointCloud ground;    // ground plane inliers (debug)
  PreprocessStats stats;
};

/// Pure C++ preprocessing pipeline — no ROS dependency, fully unit-testable.
///
/// Pipeline order (mirrors Python LidarPreprocessor exactly):
///   1. ROI crop          — AxisAlignedBoundingBox::Crop
///   2. Voxel downsample  — PointCloud::VoxelDownSample
///   3. Distance filter   — Euclidean norm <= max_depth
///   4. RANSAC ground     — PointCloud::SegmentPlane + SelectByIndex
class LidarPreprocessor {
public:
  using Config = LidarPreprocessorConfig;

  explicit LidarPreprocessor(Config cfg = Config{});

  /// Run the full 4-stage pipeline on a vector of 3-D points.
  /// Note: intensity is not passed in — voxel downsampling loses per-point
  /// intensity anyway (same behaviour as the Python implementation).
  PreprocessResult process(const std::vector<Eigen::Vector3d>& points) const;

private:
  Config cfg_;
};

}  // namespace perception_pipeline
