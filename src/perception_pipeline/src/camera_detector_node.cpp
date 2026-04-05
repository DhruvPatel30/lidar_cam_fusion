#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"

#include <opencv2/imgproc.hpp>

#include "perception_pipeline/camera_detector.hpp"

namespace perception_pipeline {

class CameraDetectorNode : public rclcpp::Node
{
public:
  CameraDetectorNode()
  : Node("camera_detector")
  {
    // ── Parameters ────────────────────────────────────────────────────────────
    declare_parameter("model_path",          std::string(""));
    declare_parameter("conf_threshold",      0.5);
    declare_parameter("nms_threshold",       0.45);
    declare_parameter("publish_debug_image", false);

    const std::string model_path =
      get_parameter("model_path").as_string();

    if (model_path.empty()) {
      RCLCPP_FATAL(get_logger(),
        "Parameter 'model_path' is required. Pass via launch: "
        "model_path:=/path/to/yolo11s.onnx  "
        "(generate with: pixi run export-yolo)");
      throw std::runtime_error("model_path not set");
    }

    CameraDetector::Config cfg;
    cfg.model_path      = model_path;
    cfg.conf_threshold  = static_cast<float>(
      get_parameter("conf_threshold").as_double());
    cfg.nms_threshold   = static_cast<float>(
      get_parameter("nms_threshold").as_double());
    publish_debug_image_ =
      get_parameter("publish_debug_image").as_bool();

    detector_ = std::make_unique<CameraDetector>(cfg);

    // ── QoS — matches kitti_publisher: RELIABLE KEEP_LAST depth 5 ─────────
    auto qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();

    // ── Subscription ─────────────────────────────────────────────────────────
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/camera/image_raw", qos,
      std::bind(&CameraDetectorNode::on_image, this, std::placeholders::_1));

    // ── Publishers ────────────────────────────────────────────────────────────
    pub_dets_ = create_publisher<vision_msgs::msg::Detection2DArray>(
      "/detections_2d", qos);

    if (publish_debug_image_) {
      auto debug_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
      pub_debug_ = create_publisher<sensor_msgs::msg::Image>(
        "/camera/detections_image", debug_qos);
    }

    RCLCPP_INFO(get_logger(),
      "CameraDetector ready — model=%s  conf=%.2f  nms=%.2f  debug=%s",
      model_path.c_str(),
      static_cast<double>(cfg.conf_threshold),
      static_cast<double>(cfg.nms_threshold),
      publish_debug_image_ ? "true" : "false");
  }

private:
  // ── Image callback ─────────────────────────────────────────────────────────
  void on_image(sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    // Decode rgb8 → BGR cv::Mat without cv_bridge
    cv::Mat bgr = image_msg_to_mat(*msg);

    std::vector<Detection> dets = detector_->detect(bgr);

    pub_dets_->publish(detections_to_msg(dets, msg->header));

    if (publish_debug_image_ && pub_debug_) {
      pub_debug_->publish(draw_detections(bgr, dets, msg->header));
    }

    ++frame_count_;
    if (frame_count_ % 20 == 0) {
      RCLCPP_INFO(get_logger(),
        "Frame %zu | detections=%zu", frame_count_, dets.size());
    }
  }

  // ── Image decode (no cv_bridge) ────────────────────────────────────────────
  static cv::Mat image_msg_to_mat(const sensor_msgs::msg::Image & msg)
  {
    // kitti_publisher publishes rgb8; wrap the buffer as a non-owning view,
    // then convert so the returned Mat owns BGR data.
    cv::Mat rgb_view(
      static_cast<int>(msg.height),
      static_cast<int>(msg.width),
      CV_8UC3,
      const_cast<uint8_t *>(msg.data.data()),
      static_cast<size_t>(msg.step));

    cv::Mat bgr;
    cv::cvtColor(rgb_view, bgr, cv::COLOR_RGB2BGR);
    return bgr;
  }

  // ── Detection → vision_msgs ────────────────────────────────────────────────
  static vision_msgs::msg::Detection2DArray detections_to_msg(
    const std::vector<Detection> & dets,
    const std_msgs::msg::Header  & header)
  {
    vision_msgs::msg::Detection2DArray arr;
    arr.header = header;
    arr.detections.reserve(dets.size());

    for (const auto & det : dets) {
      vision_msgs::msg::Detection2D d;
      d.header = header;

      // bbox: centre + size in pixel coords
      d.bbox.center.position.x = static_cast<double>(det.bbox[0] + det.bbox[2] * 0.5f);
      d.bbox.center.position.y = static_cast<double>(det.bbox[1] + det.bbox[3] * 0.5f);
      d.bbox.center.theta      = 0.0;
      d.bbox.size_x            = static_cast<double>(det.bbox[2]);
      d.bbox.size_y            = static_cast<double>(det.bbox[3]);

      vision_msgs::msg::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = det.class_name;
      hyp.hypothesis.score    = static_cast<double>(det.confidence);
      d.results.push_back(std::move(hyp));

      arr.detections.push_back(std::move(d));
    }
    return arr;
  }

  // ── Debug image ────────────────────────────────────────────────────────────
  static sensor_msgs::msg::Image draw_detections(
    const cv::Mat               & bgr,
    const std::vector<Detection>& dets,
    const std_msgs::msg::Header & header)
  {
    cv::Mat vis = bgr.clone();

    for (const auto & det : dets) {
      const int x = static_cast<int>(det.bbox[0]);
      const int y = static_cast<int>(det.bbox[1]);
      const int w = static_cast<int>(det.bbox[2]);
      const int h = static_cast<int>(det.bbox[3]);

      cv::rectangle(vis, {x, y, w, h}, {0, 255, 0}, 2);

      const std::string label =
        det.class_name + " " +
        std::to_string(static_cast<int>(det.confidence * 100)) + "%";
      cv::putText(vis, label, {x, std::max(y - 4, 0)},
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
    }

    // BGR → RGB for encoding as rgb8
    cv::Mat rgb;
    cv::cvtColor(vis, rgb, cv::COLOR_BGR2RGB);

    sensor_msgs::msg::Image out;
    out.header   = header;
    out.height   = static_cast<uint32_t>(rgb.rows);
    out.width    = static_cast<uint32_t>(rgb.cols);
    out.encoding = "rgb8";
    out.step     = static_cast<uint32_t>(rgb.cols * 3);
    out.data.assign(rgb.datastart, rgb.dataend);
    return out;
  }

  // ── Members ────────────────────────────────────────────────────────────────
  std::unique_ptr<CameraDetector> detector_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_dets_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr             pub_debug_;

  bool        publish_debug_image_ = false;
  std::size_t frame_count_         = 0;
};

}  // namespace perception_pipeline

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(
      std::make_shared<perception_pipeline::CameraDetectorNode>());
  } catch (const std::runtime_error & e) {
    // Fatal parameter error — already logged inside the constructor
    (void)e;
  }
  rclcpp::shutdown();
  return 0;
}
