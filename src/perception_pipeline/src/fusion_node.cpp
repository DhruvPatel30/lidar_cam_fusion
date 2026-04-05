#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "message_filters/subscriber.hpp"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"

#include "ament_index_cpp/get_package_share_directory.hpp"

#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc.hpp>

#include "perception_pipeline/lidar_camera_projector.hpp"

namespace perception_pipeline {

// ── PointCloud2 → Eigen helper ────────────────────────────────────────────────

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

// ── calibration loader ────────────────────────────────────────────────────────

static ProjectorConfig load_calibration(const std::string & yaml_path)
{
  YAML::Node root;
  try {
    root = YAML::LoadFile(yaml_path);
  } catch (const YAML::Exception & e) {
    throw std::runtime_error(
      std::string("Failed to load calibration file '") + yaml_path + "': " + e.what());
  }

  ProjectorConfig cfg;

  const auto T_vals = root["lidar_to_camera"]["T"].as<std::vector<double>>();
  if (T_vals.size() != 16) {
    throw std::runtime_error("calibration.yaml: lidar_to_camera.T must have 16 elements");
  }
  cfg.T = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(T_vals.data());

  const auto P_vals = root["camera"]["P"].as<std::vector<double>>();
  if (P_vals.size() != 12) {
    throw std::runtime_error("calibration.yaml: camera.P must have 12 elements");
  }
  cfg.P = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(P_vals.data());

  cfg.img_width          = root["camera"]["width"].as<int>();
  cfg.img_height         = root["camera"]["height"].as<int>();
  cfg.min_cluster_points = root["fusion"]["min_cluster_points"].as<int>();
  cfg.max_depth          = root["fusion"]["max_depth"].as<double>();

  return cfg;
}

// ── Marker color helper ───────────────────────────────────────────────────────
// Maps a class name to a stable RGBA color via hash — each class always gets
// the same color across frames.

static std_msgs::msg::ColorRGBA class_color(const std::string & class_id, float alpha = 0.35f)
{
  static const std::array<std::array<float, 3>, 8> palette = {{
    {1.0f, 0.2f, 0.2f},   // red
    {0.2f, 0.8f, 0.2f},   // green
    {0.2f, 0.4f, 1.0f},   // blue
    {1.0f, 0.8f, 0.0f},   // yellow
    {1.0f, 0.4f, 0.0f},   // orange
    {0.8f, 0.2f, 0.9f},   // purple
    {0.0f, 0.9f, 0.9f},   // cyan
    {1.0f, 0.5f, 0.8f},   // pink
  }};
  const std::size_t idx = std::hash<std::string>{}(class_id) % palette.size();
  std_msgs::msg::ColorRGBA c;
  c.r = palette[idx][0];
  c.g = palette[idx][1];
  c.b = palette[idx][2];
  c.a = alpha;
  return c;
}

// ── Depth → BGR color (jet-like) ─────────────────────────────────────────────
// norm ∈ [0, 1]: 0 = near (red), 1 = far (blue)

static cv::Scalar depth_to_bgr(float norm)
{
  // Simple 4-stop gradient: red → yellow → green → blue
  norm = std::max(0.0f, std::min(1.0f, norm));
  float r, g, b;
  if (norm < 0.33f) {
    float t = norm / 0.33f;
    r = 1.0f - t * 0.5f;   // 1 → 0.5
    g = t;                   // 0 → 1
    b = 0.0f;
  } else if (norm < 0.66f) {
    float t = (norm - 0.33f) / 0.33f;
    r = 0.5f - t * 0.5f;   // 0.5 → 0
    g = 1.0f - t * 0.5f;   // 1 → 0.5
    b = t;                   // 0 → 1
  } else {
    float t = (norm - 0.66f) / 0.34f;
    r = 0.0f;
    g = 0.5f - t * 0.5f;   // 0.5 → 0
    b = 1.0f;
  }
  return cv::Scalar(b * 255.0, g * 255.0, r * 255.0);  // BGR
}

// ── Node ──────────────────────────────────────────────────────────────────────

class FusionNode : public rclcpp::Node
{
  using PC2      = sensor_msgs::msg::PointCloud2;
  using Det2D    = vision_msgs::msg::Detection2DArray;
  using Det3DArr = vision_msgs::msg::Detection3DArray;
  using Image    = sensor_msgs::msg::Image;
  using MArr     = visualization_msgs::msg::MarkerArray;
  using Marker   = visualization_msgs::msg::Marker;
  using SyncPolicy =
    message_filters::sync_policies::ApproximateTime<PC2, Det2D>;

public:
  FusionNode()
  : Node("fusion_node")
  {
    // ── Parameters ───────────────────────────────────────────────────────────
    const std::string default_calib =
      ament_index_cpp::get_package_share_directory("perception_pipeline") +
      "/config/calibration.yaml";

    declare_parameter("calibration_file", default_calib);
    declare_parameter("sync_slop", 0.1);
    declare_parameter("publish_markers", false);
    declare_parameter("publish_debug_image", false);

    const std::string calib_path = get_parameter("calibration_file").as_string();
    const double slop            = get_parameter("sync_slop").as_double();
    publish_markers_             = get_parameter("publish_markers").as_bool();
    publish_debug_image_         = get_parameter("publish_debug_image").as_bool();

    // ── Load calibration + build projector ───────────────────────────────────
    cfg_       = load_calibration(calib_path);
    projector_ = std::make_unique<LidarCameraProjector>(cfg_);
    RCLCPP_INFO(get_logger(), "Calibration loaded from: %s", calib_path.c_str());
    RCLCPP_INFO(get_logger(), "Image size: %d × %d  min_cluster_pts: %d  max_depth: %.1f m",
      cfg_.img_width, cfg_.img_height, cfg_.min_cluster_points, cfg_.max_depth);
    RCLCPP_INFO(get_logger(), "Debug: markers=%s  debug_image=%s",
      publish_markers_ ? "ON" : "OFF",
      publish_debug_image_ ? "ON" : "OFF");

    // ── Publishers ───────────────────────────────────────────────────────────
    auto qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
    pub_      = create_publisher<Det3DArr>("/detections_3d_fused", qos);

    if (publish_markers_) {
      pub_markers_ = create_publisher<MArr>("/detections_3d_markers",
        rclcpp::QoS(rclcpp::KeepLast(5)).reliable());
    }
    if (publish_debug_image_) {
      pub_debug_image_ = create_publisher<Image>("/fusion/debug_image",
        rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
      // Subscribe at camera rate — debug image is published here, not in on_sync,
      // so it updates every frame regardless of sync frequency.
      image_sub_ = create_subscription<Image>(
        "/camera/image_raw",
        rclcpp::SensorDataQoS(),
        std::bind(&FusionNode::on_image, this, std::placeholders::_1));
    }

    // ── message_filters subscribers with ApproximateTime sync ────────────────
    const auto sub_qos = rclcpp::SensorDataQoS();
    lidar_sub_.subscribe(this, "/lidar/filtered", sub_qos.get_rmw_qos_profile());
    det_sub_.subscribe(  this, "/detections_2d",  sub_qos.get_rmw_qos_profile());

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), lidar_sub_, det_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(slop));
    sync_->registerCallback(
      std::bind(&FusionNode::on_sync, this,
        std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(),
      "FusionNode ready — sync slop %.3f s — publishing /detections_3d_fused", slop);
  }

private:
  // ── Main sync callback ───────────────────────────────────────────────────────
  void on_sync(
    const PC2::ConstSharedPtr   & pc2_msg,
    const Det2D::ConstSharedPtr & det2d_msg)
  {
    std::vector<Eigen::Vector3d> points;
    try {
      points = pc2_to_eigen(*pc2_msg);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "PointCloud2 parse error: %s", e.what());
      return;
    }
    if (points.empty()) { return; }

    auto [pixels, valid_indices] = projector_->project_to_image(points);

    std::vector<std::array<double, 4>> bboxes;
    std::vector<std::string>           class_ids;
    std::vector<double>                scores;
    for (const auto & det : det2d_msg->detections) {
      bboxes.push_back({
        det.bbox.center.position.x,
        det.bbox.center.position.y,
        det.bbox.size_x,
        det.bbox.size_y
      });
      class_ids.push_back(
        det.results.empty() ? "unknown" : det.results[0].hypothesis.class_id);
      scores.push_back(
        det.results.empty() ? 0.0 : det.results[0].hypothesis.score);
    }

    const std::vector<FusedDetection> fused =
      projector_->fuse(points, pixels, valid_indices, bboxes, class_ids, scores);

    // ── Publish Detection3DArray ─────────────────────────────────────────────
    std_msgs::msg::Header hdr;
    hdr.stamp    = pc2_msg->header.stamp;
    hdr.frame_id = "velodyne";

    Det3DArr out;
    out.header = hdr;
    for (const auto & fd : fused) {
      vision_msgs::msg::Detection3D d;
      d.header                       = hdr;
      d.bbox.center.position.x       = fd.centroid.x();
      d.bbox.center.position.y       = fd.centroid.y();
      d.bbox.center.position.z       = fd.centroid.z();
      d.bbox.center.orientation.w    = 1.0;
      d.bbox.size.x                  = fd.size.x();
      d.bbox.size.y                  = fd.size.y();
      d.bbox.size.z                  = fd.size.z();
      vision_msgs::msg::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = fd.class_id;
      hyp.hypothesis.score    = fd.score;
      d.results.push_back(hyp);
      out.detections.push_back(std::move(d));
    }
    pub_->publish(out);

    // ── Optional debug outputs ────────────────────────────────────────────────
    if (publish_markers_) {
      publish_markers_impl(fused, hdr);
    }
    // Store latest projection data so on_image() can draw at full camera rate
    if (publish_debug_image_) {
      latest_points_       = points;
      latest_pixels_       = pixels;
      latest_valid_indices_ = valid_indices;
      latest_bboxes_       = bboxes;
      latest_class_ids_    = class_ids;
      latest_scores_       = scores;
      latest_fused_        = fused;
    }

    ++frame_count_;
    if (frame_count_ % 20 == 0) {
      RCLCPP_INFO(get_logger(),
        "Frame %zu | points=%zu  projected=%zu  2d_dets=%zu  fused=%zu",
        frame_count_, points.size(), pixels.size(),
        det2d_msg->detections.size(), fused.size());
    }
  }

  // ── Image callback — runs at full camera rate ────────────────────────────────
  // Publishes the debug image on every incoming camera frame using the most
  // recently stored projection data.  Decoupled from on_sync so the debug image
  // is never throttled by sync frequency.
  void on_image(Image::ConstSharedPtr msg)
  {
    latest_image_ = msg;
    if (!publish_debug_image_ || latest_pixels_.empty()) { return; }
    publish_debug_image_impl(
      latest_points_, latest_pixels_, latest_valid_indices_,
      latest_bboxes_, latest_class_ids_, latest_scores_, latest_fused_,
      msg->header);
  }

  // ── Marker publisher ─────────────────────────────────────────────────────────
  // Publishes two markers per detection:
  //   1. CUBE  — semi-transparent box at the 3D centroid
  //   2. TEXT  — class label + confidence floating above the box
  // A DELETEALL marker is prepended so stale detections clear immediately.
  void publish_markers_impl(
    const std::vector<FusedDetection>  & fused,
    const std_msgs::msg::Header        & hdr)
  {
    MArr arr;

    // Clear all markers from the previous frame
    Marker del;
    del.header   = hdr;
    del.ns       = "fusion_detections";
    del.action   = Marker::DELETEALL;
    arr.markers.push_back(del);

    int id = 0;
    for (const auto & fd : fused) {
      const auto color = class_color(fd.class_id);

      // ── 1. Bounding box cube ───────────────────────────────────────────────
      Marker box;
      box.header        = hdr;
      box.ns            = "fusion_detections";
      box.id            = id++;
      box.type          = Marker::CUBE;
      box.action        = Marker::ADD;
      box.pose.position.x    = fd.centroid.x();
      box.pose.position.y    = fd.centroid.y();
      box.pose.position.z    = fd.centroid.z();
      box.pose.orientation.w = 1.0;
      // Clamp minimum size to 0.1 m so the marker is always visible
      box.scale.x = std::max(fd.size.x(), 0.1);
      box.scale.y = std::max(fd.size.y(), 0.1);
      box.scale.z = std::max(fd.size.z(), 0.1);
      box.color           = color;
      box.lifetime        = rclcpp::Duration::from_seconds(0.5);
      arr.markers.push_back(box);

      // ── 2. Text label ─────────────────────────────────────────────────────
      Marker txt;
      txt.header        = hdr;
      txt.ns            = "fusion_detections";
      txt.id            = id++;
      txt.type          = Marker::TEXT_VIEW_FACING;
      txt.action        = Marker::ADD;
      txt.pose.position.x    = fd.centroid.x();
      txt.pose.position.y    = fd.centroid.y();
      txt.pose.position.z    = fd.centroid.z() + std::max(fd.size.z(), 0.1) * 0.5 + 0.3;
      txt.pose.orientation.w = 1.0;
      txt.scale.z            = 0.4;     // text height in metres
      std_msgs::msg::ColorRGBA white;
      white.r = white.g = white.b = white.a = 1.0f;
      txt.color   = white;
      txt.lifetime = rclcpp::Duration::from_seconds(0.5);
      // Format: "car 92%"
      const int pct = static_cast<int>(fd.score * 100.0);
      txt.text = fd.class_id + " " + std::to_string(pct) + "%";
      arr.markers.push_back(txt);
    }

    pub_markers_->publish(arr);
  }

  // ── Debug image publisher ────────────────────────────────────────────────────
  // Draws on top of the latest /camera/image_raw frame:
  //   • All projected LiDAR points   — small circles, depth-colored (near=red, far=blue)
  //   • Points inside a fused bbox   — bright white circles (overdrawn on top)
  //   • YOLO 2D bounding boxes       — colored rectangles with class + score label
  void publish_debug_image_impl(
    const std::vector<Eigen::Vector3d>        & points,
    const std::vector<Eigen::Vector2d>        & pixels,
    const std::vector<std::size_t>            & valid_indices,
    const std::vector<std::array<double, 4>>  & bboxes,
    const std::vector<std::string>            & class_ids,
    const std::vector<double>                 & scores,
    const std::vector<FusedDetection>         & fused,
    const std_msgs::msg::Header               & /*hdr*/)
  {
    // Decode rgb8 image → BGR cv::Mat (same pattern as camera_detector_node)
    const auto & img = *latest_image_;  // latest_image_ is set in on_image before this call
    cv::Mat rgb_view(
      static_cast<int>(img.height),
      static_cast<int>(img.width),
      CV_8UC3,
      const_cast<uint8_t *>(img.data.data()),
      static_cast<std::size_t>(img.step));
    cv::Mat vis;
    cv::cvtColor(rgb_view, vis, cv::COLOR_RGB2BGR);

    // ── Draw all projected LiDAR points (depth-colored) ──────────────────────
    for (std::size_t i = 0; i < pixels.size(); ++i) {
      const double dist = points[valid_indices[i]].norm();
      const float  norm = static_cast<float>(dist / cfg_.max_depth);
      const int    u    = static_cast<int>(std::round(pixels[i].x()));
      const int    v    = static_cast<int>(std::round(pixels[i].y()));
      cv::circle(vis, {u, v}, 2, depth_to_bgr(norm), -1);
    }

    // ── Highlight points that are inside a fused detection bbox ──────────────
    for (std::size_t b = 0; b < bboxes.size(); ++b) {
      // Only highlight bboxes that produced a fused detection
      bool has_fused = false;
      for (const auto & fd : fused) {
        if (fd.class_id == class_ids[b]) { has_fused = true; break; }
      }
      if (!has_fused) { continue; }

      const double x_min = bboxes[b][0] - bboxes[b][2] * 0.5;
      const double x_max = bboxes[b][0] + bboxes[b][2] * 0.5;
      const double y_min = bboxes[b][1] - bboxes[b][3] * 0.5;
      const double y_max = bboxes[b][1] + bboxes[b][3] * 0.5;

      for (std::size_t i = 0; i < pixels.size(); ++i) {
        const double u = pixels[i].x();
        const double v = pixels[i].y();
        if (u >= x_min && u <= x_max && v >= y_min && v <= y_max) {
          cv::circle(vis, {static_cast<int>(u), static_cast<int>(v)},
            3, cv::Scalar(255, 255, 255), -1);
        }
      }
    }

    // ── Draw YOLO 2D bounding boxes ───────────────────────────────────────────
    for (std::size_t b = 0; b < bboxes.size(); ++b) {
      const int x = static_cast<int>(bboxes[b][0] - bboxes[b][2] * 0.5);
      const int y = static_cast<int>(bboxes[b][1] - bboxes[b][3] * 0.5);
      const int w = static_cast<int>(bboxes[b][2]);
      const int h = static_cast<int>(bboxes[b][3]);

      // Use the same class color as the marker (convert to BGR)
      const auto rc = class_color(class_ids[b], 1.0f);
      const cv::Scalar box_color(rc.b * 255, rc.g * 255, rc.r * 255);
      cv::rectangle(vis, {x, y, w, h}, box_color, 2);

      const int pct = static_cast<int>(scores[b] * 100.0);
      const std::string label = class_ids[b] + " " + std::to_string(pct) + "%";
      cv::putText(vis, label, {x, std::max(y - 4, 0)},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);
    }

    // ── Encode BGR → RGB and publish ─────────────────────────────────────────
    cv::Mat rgb_out;
    cv::cvtColor(vis, rgb_out, cv::COLOR_BGR2RGB);

    sensor_msgs::msg::Image out;
    out.header   = latest_image_->header;
    out.height   = static_cast<uint32_t>(rgb_out.rows);
    out.width    = static_cast<uint32_t>(rgb_out.cols);
    out.encoding = "rgb8";
    out.step     = static_cast<uint32_t>(rgb_out.cols * 3);
    out.data.assign(rgb_out.datastart, rgb_out.dataend);
    pub_debug_image_->publish(out);
  }

  // ── Members ──────────────────────────────────────────────────────────────────
  ProjectorConfig                                          cfg_;
  std::unique_ptr<LidarCameraProjector>                   projector_;

  message_filters::Subscriber<PC2>                        lidar_sub_;
  message_filters::Subscriber<Det2D>                      det_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Publisher<Det3DArr>::SharedPtr                  pub_;
  rclcpp::Publisher<MArr>::SharedPtr                      pub_markers_;
  rclcpp::Publisher<Image>::SharedPtr                     pub_debug_image_;
  rclcpp::Subscription<Image>::SharedPtr                  image_sub_;
  Image::ConstSharedPtr                                   latest_image_;

  bool        publish_markers_     = false;
  bool        publish_debug_image_ = false;
  std::size_t frame_count_         = 0;

  // Latest projection state — written by on_sync, read by on_image
  std::vector<Eigen::Vector3d>       latest_points_;
  std::vector<Eigen::Vector2d>       latest_pixels_;
  std::vector<std::size_t>           latest_valid_indices_;
  std::vector<std::array<double, 4>> latest_bboxes_;
  std::vector<std::string>           latest_class_ids_;
  std::vector<double>                latest_scores_;
  std::vector<FusedDetection>        latest_fused_;
};

}  // namespace perception_pipeline

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<perception_pipeline::FusionNode>());
  rclcpp::shutdown();
  return 0;
}
