"""
KITTI Publisher Node
====================
Reads raw KITTI dataset files from disk and republishes them as ROS 2 topics,
simulating a live sensor setup for the rest of the fusion pipeline.

Published topics:
  /camera/image_raw   sensor_msgs/Image        (left color camera, image_02)
  /lidar/points       sensor_msgs/PointCloud2  (Velodyne HDL-64, float32 x/y/z/intensity)

Parameters:
  sequence_path  (str)   Absolute path to a KITTI raw sync sequence directory.
                         e.g. /data/kitti/2011_09_26/2011_09_26_drive_0001_sync
  frame_rate     (float) Playback rate in Hz. Default: 10.0
  loop           (bool)  Restart from frame 0 after the last frame. Default: True
  camera_id      (str)   Which camera folder to read. Default: image_02 (left color)

Usage:
  pixi run ros2 run perception_pipeline kitti_publisher \
      --ros-args -p sequence_path:=/path/to/sequence -p frame_rate:=10.0
"""

import struct
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header


# ── PointCloud2 helper (avoids cv_bridge / sensor_msgs_py version skew) ───────

_POINTCLOUD_DTYPE = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("intensity", np.float32),
])


def _bin_to_pointcloud2(bin_path: Path, header: Header) -> PointCloud2:
    """Convert a KITTI .bin point cloud file to a sensor_msgs/PointCloud2 message.

    KITTI Velodyne format: binary float32 array, each point = [x, y, z, intensity].
    """
    raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = raw.shape[0]
    msg.is_bigendian = False
    msg.is_dense = True

    itemsize = np.dtype(np.float32).itemsize  # 4 bytes
    msg.fields = [
        PointField(name="x",         offset=0 * itemsize, datatype=PointField.FLOAT32, count=1),
        PointField(name="y",         offset=1 * itemsize, datatype=PointField.FLOAT32, count=1),
        PointField(name="z",         offset=2 * itemsize, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=3 * itemsize, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 4 * itemsize   # 16 bytes per point
    msg.row_step = msg.point_step * msg.width
    msg.data = raw.tobytes()
    return msg


def _png_to_image(png_path: Path, header: Header) -> Image:
    """Convert a PNG file to a sensor_msgs/Image message (rgb8 encoding).

    Does not require cv_bridge — converts directly via numpy.
    """
    bgr = cv2.imread(str(png_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {png_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    msg = Image()
    msg.header = header
    msg.height = rgb.shape[0]
    msg.width = rgb.shape[1]
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = rgb.shape[1] * 3
    msg.data = rgb.tobytes()
    return msg


# ── Node ──────────────────────────────────────────────────────────────────────

class KittiPublisherNode(Node):
    def __init__(self):
        super().__init__("kitti_publisher")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("sequence_path", "")
        self.declare_parameter("frame_rate", 10.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("camera_id", "image_02")

        seq_path_str = self.get_parameter("sequence_path").get_parameter_value().string_value
        self._frame_rate = self.get_parameter("frame_rate").get_parameter_value().double_value
        self._loop = self.get_parameter("loop").get_parameter_value().bool_value
        camera_id = self.get_parameter("camera_id").get_parameter_value().string_value

        if not seq_path_str:
            self.get_logger().fatal(
                "Parameter 'sequence_path' is required. "
                "Pass --ros-args -p sequence_path:=/path/to/kitti/sequence"
            )
            raise RuntimeError("sequence_path not set")

        self._seq_path = Path(seq_path_str)
        self._validate_sequence_dir(camera_id)

        # ── File lists (sorted = chronological) ───────────────────────────────
        self._image_files = sorted(
            (self._seq_path / camera_id / "data").glob("*.png")
        )
        self._lidar_files = sorted(
            (self._seq_path / "velodyne_points" / "data").glob("*.bin")
        )

        n_img = len(self._image_files)
        n_lid = len(self._lidar_files)
        if n_img == 0 or n_lid == 0:
            raise RuntimeError(
                f"No data found in sequence: images={n_img}, lidar={n_lid}"
            )

        # KITTI sequences are always paired; warn if counts differ
        if n_img != n_lid:
            self.get_logger().warn(
                f"Frame count mismatch: {n_img} images vs {n_lid} lidar scans. "
                "Will publish min(n_img, n_lid) frames."
            )

        self._n_frames = min(n_img, n_lid)
        self._frame_idx = 0

        # ── QoS — best-effort, small queue (mirrors live sensor drivers) ──────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self._img_pub = self.create_publisher(Image, "/camera/image_raw", qos)
        self._pc_pub = self.create_publisher(PointCloud2, "/lidar/points", qos)

        # ── Timer ─────────────────────────────────────────────────────────────
        period = 1.0 / self._frame_rate
        self._timer = self.create_timer(period, self._publish_frame)

        self.get_logger().info(
            f"KittiPublisher ready — {self._n_frames} frames @ {self._frame_rate} Hz"
            f"  loop={self._loop}  seq={self._seq_path.name}"
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _validate_sequence_dir(self, camera_id: str) -> None:
        """Check that the expected KITTI subdirectories exist."""
        required = [
            self._seq_path / camera_id / "data",
            self._seq_path / "velodyne_points" / "data",
        ]
        for d in required:
            if not d.is_dir():
                raise RuntimeError(
                    f"Expected directory not found: {d}\n"
                    "Ensure sequence_path points to a KITTI raw sync sequence, e.g.:\n"
                    "  2011_09_26/2011_09_26_drive_0001_sync/"
                )

    def _make_header(self, frame_id: str) -> Header:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        return header

    def _publish_frame(self) -> None:
        if self._frame_idx >= self._n_frames:
            if self._loop:
                self._frame_idx = 0
                self.get_logger().info("Looping back to frame 0")
            else:
                self.get_logger().info("Sequence finished. Shutting down.")
                self._timer.cancel()
                rclpy.shutdown()
                return

        idx = self._frame_idx
        stamp = self.get_clock().now().to_msg()

        # Publish image
        try:
            img_header = Header()
            img_header.stamp = stamp
            img_header.frame_id = "camera_left"
            img_msg = _png_to_image(self._image_files[idx], img_header)
            self._img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Image frame {idx}: {e}")

        # Publish point cloud
        try:
            pc_header = Header()
            pc_header.stamp = stamp      # same stamp → fused downstream
            pc_header.frame_id = "velodyne"
            pc_msg = _bin_to_pointcloud2(self._lidar_files[idx], pc_header)
            self._pc_pub.publish(pc_msg)
        except Exception as e:
            self.get_logger().error(f"LiDAR frame {idx}: {e}")

        if idx % 50 == 0:
            self.get_logger().info(f"Publishing frame {idx}/{self._n_frames - 1}")

        self._frame_idx += 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    try:
        node = KittiPublisherNode()
        rclpy.spin(node)
    except RuntimeError as e:
        print(f"[kitti_publisher] Fatal: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
