#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include "perception_pipeline/lidar_preprocessor.hpp"

namespace perception_pipeline {

// ── PointCloud2 ↔ Eigen helpers ───────────────────────────────────────────────

/// Convert sensor_msgs/PointCloud2 → vector of Eigen::Vector3d.
/// Uses PointCloud2ConstIterator — field-layout agnostic, reads x/y/z by name.
static std::vector<Eigen::Vector3d>
pc2_to_eigen(const sensor_msgs::msg::PointCloud2 & msg)
{
  sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");

  const std::size_t n = msg.width * msg.height;
  std::vector<Eigen::Vector3d> pts;
  pts.reserve(n);
  for (std::size_t i = 0; i < n; ++i, ++iter_x, ++iter_y, ++iter_z) {
    pts.emplace_back(
      static_cast<double>(*iter_x),
      static_cast<double>(*iter_y),
      static_cast<double>(*iter_z));
  }
  return pts;
}

/// Convert Open3D PointCloud → sensor_msgs/PointCloud2.
/// Outputs x, y, z, intensity (intensity = 0 — voxel downsampling loses
/// per-point intensity, same behaviour as the Python node).
static sensor_msgs::msg::PointCloud2
eigen_to_pc2(
  const open3d::geometry::PointCloud & pcd,
  const std_msgs::msg::Header & header)
{
  sensor_msgs::msg::PointCloud2 msg;
  msg.header = header;
  msg.height = 1;
  msg.width  = static_cast<uint32_t>(pcd.points_.size());
  msg.is_bigendian = false;
  msg.is_dense = true;

  sensor_msgs::PointCloud2Modifier modifier(msg);
  modifier.setPointCloud2Fields(
    4,
    "x",         1, sensor_msgs::msg::PointField::FLOAT32,
    "y",         1, sensor_msgs::msg::PointField::FLOAT32,
    "z",         1, sensor_msgs::msg::PointField::FLOAT32,
    "intensity", 1, sensor_msgs::msg::PointField::FLOAT32
  );
  modifier.resize(pcd.points_.size());

  sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");
  sensor_msgs::PointCloud2Iterator<float> iter_i(msg, "intensity");

  for (const auto & pt : pcd.points_) {
    *iter_x = static_cast<float>(pt.x());
    *iter_y = static_cast<float>(pt.y());
    *iter_z = static_cast<float>(pt.z());
    *iter_i = 0.0f;
    ++iter_x; ++iter_y; ++iter_z; ++iter_i;
  }
  return msg;
}

// ── Node ──────────────────────────────────────────────────────────────────────

class LidarProcessorNode : public rclcpp::Node
{
public:
  LidarProcessorNode()
  : Node("lidar_processor")
  {
    // ── Parameters (same names and defaults as Python node) ──────────────────
    declare_parameter("roi_x_min",    0.0);
    declare_parameter("roi_x_max",   50.0);
    declare_parameter("roi_y_min",  -10.0);
    declare_parameter("roi_y_max",   10.0);
    declare_parameter("roi_z_min",   -3.0);
    declare_parameter("roi_z_max",    2.0);
    declare_parameter("voxel_size",   0.1);
    declare_parameter("ransac_dist",  0.2);
    declare_parameter("ransac_iter", 100);
    declare_parameter("max_depth",   50.0);

    LidarPreprocessor::Config cfg;
    cfg.roi_x_min   = get_parameter("roi_x_min").as_double();
    cfg.roi_x_max   = get_parameter("roi_x_max").as_double();
    cfg.roi_y_min   = get_parameter("roi_y_min").as_double();
    cfg.roi_y_max   = get_parameter("roi_y_max").as_double();
    cfg.roi_z_min   = get_parameter("roi_z_min").as_double();
    cfg.roi_z_max   = get_parameter("roi_z_max").as_double();
    cfg.voxel_size  = get_parameter("voxel_size").as_double();
    cfg.ransac_dist = get_parameter("ransac_dist").as_double();
    cfg.ransac_iter = get_parameter("ransac_iter").as_int();
    cfg.max_depth   = get_parameter("max_depth").as_double();

    preprocessor_ = std::make_unique<LidarPreprocessor>(cfg);

    // ── QoS — matches Python node: RELIABLE, KEEP_LAST depth 5 ─────────────
    auto qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/lidar/points", qos,
      std::bind(&LidarProcessorNode::on_pointcloud, this, std::placeholders::_1));

    pub_filtered_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/lidar/filtered", qos);
    pub_ground_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/lidar/ground_plane", qos);

    RCLCPP_INFO(get_logger(),
      "LidarProcessor ready — C++ / Open3D — waiting for /lidar/points");
  }

private:
  void on_pointcloud(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    std::vector<Eigen::Vector3d> pts;
    try {
      pts = pc2_to_eigen(*msg);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "PointCloud2 parse error: %s", e.what());
      return;
    }

    PreprocessResult result;
    try {
      result = preprocessor_->process(pts);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Preprocessing error on frame %zu: %s",
        frame_count_, e.what());
      return;
    }

    pub_filtered_->publish(eigen_to_pc2(result.filtered, msg->header));
    pub_ground_->publish(eigen_to_pc2(result.ground,   msg->header));

    ++frame_count_;
    if (frame_count_ % 20 == 0) {
      const auto & s = result.stats;
      RCLCPP_INFO(get_logger(),
        "Frame %zu | in=%6zu  roi=%6zu  voxel=%6zu  ground=%5zu  out=%6zu",
        frame_count_, s.n_input, s.n_roi, s.n_voxel, s.n_ground, s.n_output);
    }
  }

  std::unique_ptr<LidarPreprocessor> preprocessor_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_filtered_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_;
  std::size_t frame_count_ = 0;
};

}  // namespace perception_pipeline

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<perception_pipeline::LidarProcessorNode>());
  rclcpp::shutdown();
  return 0;
}
