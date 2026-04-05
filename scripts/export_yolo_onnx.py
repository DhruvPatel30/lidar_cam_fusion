#!/usr/bin/env python3
"""Export YOLOv11s to ONNX format for the C++ camera_detector node.

Usage:
    pixi run export-yolo

Output:
    models/yolo11s.onnx  (relative to workspace root)

The exported model has:
  input  shape: [1, 3, 640, 640]   (NCHW, RGB, normalised to [0, 1])
  output shape: [1, 84, 8400]      (4 box coords + 80 COCO class scores,
                                    8400 anchors for 640x640 input)
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent.parent
MODELS_DIR = WORKSPACE_ROOT / "models"
OUTPUT_PATH = MODELS_DIR / "yolo11s.onnx"


def main() -> None:
    from ultralytics import YOLO  # imported here so the script fails fast if missing

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading yolo11s.pt (downloads automatically on first run)...")
    model = YOLO("yolo11s.pt")

    print("Exporting to ONNX (opset 12, static batch=1, imgsz=640)...")
    model.export(
        format="onnx",
        opset=12,
        simplify=True,
        imgsz=640,
        dynamic=False,  # fixed batch=1 — simplifies the C++ API
    )

    # ultralytics writes <stem>.onnx next to the .pt or in cwd depending on version
    for candidate in [Path("yolo11s.onnx"), Path("yolo11s") / "yolo11s.onnx"]:
        if candidate.exists() and candidate.resolve() != OUTPUT_PATH.resolve():
            candidate.rename(OUTPUT_PATH)
            break

    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Export succeeded but could not locate output file. "
            f"Expected: {OUTPUT_PATH}"
        )

    print(f"\nExported: {OUTPUT_PATH.resolve()}")
    print("Pass this path to the camera_detector node via:")
    print(f"  model_path:={OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
