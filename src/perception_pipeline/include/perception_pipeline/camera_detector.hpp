#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace perception_pipeline {

// ── Configuration ─────────────────────────────────────────────────────────────

struct CameraDetectorConfig {
  std::string model_path;           ///< Absolute path to the ONNX model file.
  float conf_threshold = 0.5f;      ///< Minimum class score to keep a detection.
  float nms_threshold  = 0.45f;     ///< IoU threshold for non-maximum suppression.
  int   input_width    = 640;       ///< Model input width  (must match export).
  int   input_height   = 640;       ///< Model input height (must match export).
};

// ── Result types ──────────────────────────────────────────────────────────────

/// Single 2-D object detection in original image pixel coordinates.
struct Detection {
  /// Bounding box [x, y, w, h]: x/y = top-left corner, w/h = size (pixels).
  std::array<float, 4> bbox;
  int         class_id;
  std::string class_name;
  float       confidence;
};

// ── Detector class ────────────────────────────────────────────────────────────

/// Pure C++ YOLOv11 detector using ONNX Runtime inference.
///
/// No ROS dependency — fully unit-testable in isolation.
///
/// Expected ONNX model (exported with ultralytics):
///   input  shape: [1, 3, 640, 640]  (NCHW, RGB float32, normalised to [0,1])
///   output shape: [1, 84, 8400]     (rows 0-3: cx,cy,w,h; rows 4-83: 80 COCO scores)
///
/// Usage:
///   CameraDetector det({.model_path = "/path/to/yolo11s.onnx"});
///   std::vector<Detection> dets = det.detect(bgr_image);
class CameraDetector {
public:
  using Config = CameraDetectorConfig;

  explicit CameraDetector(Config cfg);
  ~CameraDetector();  // defined in .cpp so unique_ptr<OrtState> compiles

  /// Run inference on a BGR image (CV_8UC3).
  /// Returns detections in original image pixel coordinates.
  /// Thread-safe after construction (stateless per call).
  std::vector<Detection> detect(const cv::Mat & bgr_image) const;

  // Non-copyable (holds a live ONNX session).
  CameraDetector(const CameraDetector &)            = delete;
  CameraDetector & operator=(const CameraDetector &) = delete;

private:
  // ── Pimpl: hides all ONNX Runtime ABI from this header ────────────────────
  struct OrtState;
  std::unique_ptr<OrtState> ort_;

  Config cfg_;

  // ── Constants ─────────────────────────────────────────────────────────────
  static constexpr int kNumClasses = 80;
  static constexpr int kNumAnchors = 8400;
  static constexpr int kOutputRows = 84;   // 4 box + 80 class scores

  /// COCO 2017 class names in index order (0 = person, 79 = toothbrush).
  static const std::array<std::string, 80> kCocoNames;

  // ── Pre-processing helpers ─────────────────────────────────────────────────

  /// Resize src to cfg_.input_width × cfg_.input_height with letterboxing.
  /// Fills padding with grey (114, 114, 114) — matches ultralytics default.
  /// Outputs the uniform scale factor and integer padding offsets used to
  /// unproject detections back to original image coordinates.
  cv::Mat letterbox(const cv::Mat & src,
                    float & scale,
                    int   & pad_x,
                    int   & pad_y) const;

  /// Convert a 640×640 RGB CV_8UC3 mat to a flat NCHW float32 vector
  /// normalised to [0, 1].  Layout: all R-plane, then G-plane, then B-plane.
  std::vector<float> mat_to_nchw_float(const cv::Mat & rgb_letterboxed) const;

  // ── Post-processing helpers ────────────────────────────────────────────────

  /// Decode the raw [1, 84, 8400] output tensor into Detection objects.
  /// scale / pad_x / pad_y are the values returned by letterbox().
  std::vector<Detection> decode_output(const float * data,
                                       float scale,
                                       int   pad_x,
                                       int   pad_y,
                                       int   orig_w,
                                       int   orig_h) const;

  /// Greedy IoU-based non-maximum suppression.
  /// boxes: [x1, y1, x2, y2] format.  Returns indices of kept boxes.
  static std::vector<int> nms(const std::vector<std::array<float, 4>> & boxes,
                               const std::vector<float> & scores,
                               float iou_threshold);

  /// Intersection-over-union for two [x1,y1,x2,y2] boxes.
  static float iou(const std::array<float, 4> & a,
                   const std::array<float, 4> & b);
};

}  // namespace perception_pipeline
