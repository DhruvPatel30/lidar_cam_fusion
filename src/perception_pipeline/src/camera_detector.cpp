#include "perception_pipeline/camera_detector.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

#include <core/session/onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

namespace perception_pipeline {

// ── COCO class names ──────────────────────────────────────────────────────────

const std::array<std::string, 80> CameraDetector::kCocoNames = {
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
  "truck", "boat", "traffic light", "fire hydrant", "stop sign",
  "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
  "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
  "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
  "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
  "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
  "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

// ── OrtState pimpl ────────────────────────────────────────────────────────────

struct CameraDetector::OrtState {
  Ort::Env            env;
  Ort::SessionOptions session_options;
  Ort::Session        session;
  std::string         input_name;
  std::string         output_name;

  OrtState(const std::string & model_path)
  : env(ORT_LOGGING_LEVEL_WARNING, "camera_detector"),
    session_options(),
    session(nullptr)
  {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session = Ort::Session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    input_name  = std::string(session.GetInputNameAllocated(0,  allocator).get());
    output_name = std::string(session.GetOutputNameAllocated(0, allocator).get());
  }
};

// ── Constructor / destructor ──────────────────────────────────────────────────

CameraDetector::CameraDetector(Config cfg)
: cfg_(std::move(cfg))
{
  if (cfg_.model_path.empty()) {
    throw std::runtime_error("CameraDetector: model_path must not be empty");
  }

  ort_ = std::make_unique<OrtState>(cfg_.model_path);

  // Validate output tensor shape: expect [1, 84, 8400]
  auto out_info  = ort_->session.GetOutputTypeInfo(0);
  auto out_shape = out_info.GetTensorTypeAndShapeInfo().GetShape();
  if (out_shape.size() != 3 ||
      out_shape[1] != kOutputRows ||
      out_shape[2] != kNumAnchors)
  {
    throw std::runtime_error(
      "CameraDetector: unexpected output shape. "
      "Expected [1, 84, 8400], got [" +
      std::to_string(out_shape[0]) + ", " +
      std::to_string(out_shape[1]) + ", " +
      std::to_string(out_shape[2]) + "]. "
      "Re-export with: model.export(format='onnx', imgsz=640, dynamic=False)");
  }
}

CameraDetector::~CameraDetector() = default;

// ── Pre-processing ────────────────────────────────────────────────────────────

cv::Mat CameraDetector::letterbox(const cv::Mat & src,
                                   float & scale,
                                   int   & pad_x,
                                   int   & pad_y) const
{
  const float r_w = static_cast<float>(cfg_.input_width)  / src.cols;
  const float r_h = static_cast<float>(cfg_.input_height) / src.rows;
  scale = std::min(r_w, r_h);

  const int new_w = static_cast<int>(std::round(src.cols * scale));
  const int new_h = static_cast<int>(std::round(src.rows * scale));
  pad_x = (cfg_.input_width  - new_w) / 2;
  pad_y = (cfg_.input_height - new_h) / 2;

  cv::Mat resized;
  cv::resize(src, resized, {new_w, new_h}, 0.0, 0.0, cv::INTER_LINEAR);

  cv::Mat out;
  cv::copyMakeBorder(
    resized, out,
    pad_y, cfg_.input_height - new_h - pad_y,
    pad_x, cfg_.input_width  - new_w - pad_x,
    cv::BORDER_CONSTANT,
    cv::Scalar(114, 114, 114));  // grey fill — matches ultralytics default

  return out;
}

std::vector<float> CameraDetector::mat_to_nchw_float(
  const cv::Mat & rgb_lb) const
{
  const int h = cfg_.input_height;
  const int w = cfg_.input_width;
  const int plane = h * w;

  std::vector<float> data(3 * plane);

  std::vector<cv::Mat> channels(3);
  cv::split(rgb_lb, channels);

  for (int c = 0; c < 3; ++c) {
    const cv::Mat & ch = channels[c];
    float * dst = data.data() + c * plane;
    for (int i = 0; i < plane; ++i) {
      dst[i] = ch.data[i] / 255.0f;
    }
  }
  return data;
}

// ── Inference ─────────────────────────────────────────────────────────────────

std::vector<Detection> CameraDetector::detect(const cv::Mat & bgr_image) const
{
  if (bgr_image.empty()) return {};

  const int orig_w = bgr_image.cols;
  const int orig_h = bgr_image.rows;

  // 1. BGR → RGB (YOLO was trained on RGB)
  cv::Mat rgb;
  cv::cvtColor(bgr_image, rgb, cv::COLOR_BGR2RGB);

  // 2. Letterbox to model input size
  float scale;
  int   pad_x, pad_y;
  cv::Mat lb = letterbox(rgb, scale, pad_x, pad_y);

  // 3. HWC uint8 → NCHW float32 [0,1]
  std::vector<float> input_data = mat_to_nchw_float(lb);

  // 4. Build ONNX input tensor
  Ort::MemoryInfo mem_info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  int64_t input_shape[] = {1, 3,
                            static_cast<int64_t>(cfg_.input_height),
                            static_cast<int64_t>(cfg_.input_width)};

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    mem_info,
    input_data.data(), input_data.size(),
    input_shape, 4);

  const char * input_names[]  = {ort_->input_name.c_str()};
  const char * output_names[] = {ort_->output_name.c_str()};

  // 5. Run inference
  auto outputs = ort_->session.Run(
    Ort::RunOptions{nullptr},
    input_names,  &input_tensor, 1,
    output_names, 1);

  const float * out_data = outputs[0].GetTensorData<float>();

  // 6. Decode output tensor → Detection objects
  return decode_output(out_data, scale, pad_x, pad_y, orig_w, orig_h);
}

// ── Post-processing ───────────────────────────────────────────────────────────

std::vector<Detection> CameraDetector::decode_output(
  const float * data,
  float scale,
  int   pad_x,
  int   pad_y,
  int   orig_w,
  int   orig_h) const
{
  // Output layout: data[row * kNumAnchors + anchor]
  // row 0-3  : cx, cy, w, h  (in letterboxed 640×640 pixel space)
  // row 4-83 : class scores  (raw values; max is the confidence)

  // Per-class candidate storage for NMS
  struct Candidate {
    std::array<float, 4> xyxy;  // [x1,y1,x2,y2] in original image space
    std::array<float, 4> xywh;  // [x,y,w,h]   in original image space
    float score;
    int   class_id;
  };

  std::vector<std::vector<Candidate>> per_class(kNumClasses);

  for (int a = 0; a < kNumAnchors; ++a) {
    // Find max class score and its index
    float max_score = -1.0f;
    int   best_cls  = 0;
    for (int c = 0; c < kNumClasses; ++c) {
      const float s = data[(4 + c) * kNumAnchors + a];
      if (s > max_score) { max_score = s; best_cls = c; }
    }

    if (max_score < cfg_.conf_threshold) continue;

    // Centre-format box in letterboxed 640×640 space
    const float cx = data[0 * kNumAnchors + a];
    const float cy = data[1 * kNumAnchors + a];
    const float bw = data[2 * kNumAnchors + a];
    const float bh = data[3 * kNumAnchors + a];

    // Convert to corner format, unpad, unscale → original image pixels
    auto unproject = [&](float v_lb, float pad, float sc, float limit) -> float {
      return std::clamp((v_lb - pad) / sc, 0.0f, static_cast<float>(limit));
    };

    const float x1 = unproject(cx - bw * 0.5f, static_cast<float>(pad_x), scale, orig_w);
    const float y1 = unproject(cy - bh * 0.5f, static_cast<float>(pad_y), scale, orig_h);
    const float x2 = unproject(cx + bw * 0.5f, static_cast<float>(pad_x), scale, orig_w);
    const float y2 = unproject(cy + bh * 0.5f, static_cast<float>(pad_y), scale, orig_h);

    Candidate cand;
    cand.xyxy     = {x1, y1, x2, y2};
    cand.xywh     = {x1, y1, x2 - x1, y2 - y1};
    cand.score    = max_score;
    cand.class_id = best_cls;

    per_class[best_cls].push_back(cand);
  }

  // Per-class NMS
  std::vector<Detection> results;
  for (int c = 0; c < kNumClasses; ++c) {
    auto & cands = per_class[c];
    if (cands.empty()) continue;

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    boxes.reserve(cands.size());
    scores.reserve(cands.size());
    for (const auto & cd : cands) {
      boxes.push_back(cd.xyxy);
      scores.push_back(cd.score);
    }

    for (int idx : nms(boxes, scores, cfg_.nms_threshold)) {
      Detection det;
      det.bbox       = cands[idx].xywh;
      det.class_id   = c;
      det.class_name = kCocoNames[static_cast<std::size_t>(c)];
      det.confidence = cands[idx].score;
      results.push_back(std::move(det));
    }
  }

  return results;
}

// ── NMS ───────────────────────────────────────────────────────────────────────

std::vector<int> CameraDetector::nms(
  const std::vector<std::array<float, 4>> & boxes,
  const std::vector<float> & scores,
  float iou_threshold)
{
  // Sort indices by score descending
  std::vector<int> order(boxes.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
    [&](int i, int j) { return scores[i] > scores[j]; });

  std::vector<bool> suppressed(boxes.size(), false);
  std::vector<int>  kept;

  for (std::size_t i = 0; i < order.size(); ++i) {
    const int idx = order[i];
    if (suppressed[idx]) continue;
    kept.push_back(idx);
    for (std::size_t j = i + 1; j < order.size(); ++j) {
      const int jdx = order[j];
      if (!suppressed[jdx] && iou(boxes[idx], boxes[jdx]) > iou_threshold) {
        suppressed[jdx] = true;
      }
    }
  }
  return kept;
}

float CameraDetector::iou(const std::array<float, 4> & a,
                           const std::array<float, 4> & b)
{
  // a, b are [x1, y1, x2, y2]
  const float ix1 = std::max(a[0], b[0]);
  const float iy1 = std::max(a[1], b[1]);
  const float ix2 = std::min(a[2], b[2]);
  const float iy2 = std::min(a[3], b[3]);

  const float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
  if (inter == 0.0f) return 0.0f;

  const float area_a = (a[2] - a[0]) * (a[3] - a[1]);
  const float area_b = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (area_a + area_b - inter);
}

}  // namespace perception_pipeline
