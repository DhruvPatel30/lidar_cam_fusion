#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace perception_pipeline {

/// Configuration loaded once from calibration.yaml.
struct ProjectorConfig {
  Eigen::Matrix4d             T;                  ///< 4×4 Velodyne → rectified camera frame
  Eigen::Matrix<double, 3, 4> P;                  ///< 3×4 projection matrix (incorporates rectification)
  int    img_width          = 1242;
  int    img_height         = 375;
  int    min_cluster_points = 5;
  double max_depth          = 50.0;
};

/// One fused detection: a 3-D bounding box in the Velodyne frame with its
/// associated YOLO class and confidence.
struct FusedDetection {
  Eigen::Vector3d centroid;        ///< mean of matched LiDAR points (Velodyne frame)
  Eigen::Vector3d size;            ///< axis-aligned extent (max − min per axis)
  std::string     class_id;
  double          score = 0.0;
};

/// Pure C++ projection + frustum-association logic.  No ROS dependency —
/// mirrors the library pattern of LidarPreprocessor and CameraDetector.
class LidarCameraProjector
{
public:
  explicit LidarCameraProjector(const ProjectorConfig & cfg);

  /// Project Velodyne points onto the image plane.
  ///
  /// For each point p = (x, y, z):
  ///   p_rect = T * [x, y, z, 1]^T          (Velodyne → rectified camera)
  ///   p_img  = P * p_rect                   (3×4 projection → homogeneous)
  ///   u = p_img[0] / p_img[2],  v = p_img[1] / p_img[2]
  ///
  /// Returns {valid_pixels (M×2), valid_indices (M)} — valid_indices[i] is the
  /// index into `points` that produced pixel valid_pixels[i].
  std::pair<std::vector<Eigen::Vector2d>, std::vector<std::size_t>>
  project_to_image(const std::vector<Eigen::Vector3d> & points) const;

  /// For each 2-D bounding box {cx, cy, w, h} find which projected LiDAR
  /// pixels land inside it, retrieve their 3-D Velodyne coordinates, fit an
  /// axis-aligned 3-D box, and return a FusedDetection per successful match.
  /// Boxes with fewer than min_cluster_points hits are skipped.
  std::vector<FusedDetection> fuse(
    const std::vector<Eigen::Vector3d> &        points,
    const std::vector<Eigen::Vector2d> &        pixels,
    const std::vector<std::size_t> &            valid_indices,
    const std::vector<std::array<double, 4>> &  bboxes,      ///< {cx, cy, w, h}
    const std::vector<std::string> &            class_ids,
    const std::vector<double> &                 scores) const;

private:
  ProjectorConfig cfg_;
};

}  // namespace perception_pipeline
