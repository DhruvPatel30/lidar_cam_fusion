"""
Step 3 verification — LiDAR Processor Node
Run with: pixi run verify3
"""

import sys
import time
import tempfile
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src/perception_pipeline"))

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
WARN = "\033[93m  WARN\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"
failures = []


def check(label, ok, detail="", warn_only=False):
    tag = (WARN if warn_only else FAIL) if not ok else PASS
    suffix = f"  ({detail})" if detail else ""
    print(f"{tag}  {label}{suffix}")
    if not ok and not warn_only:
        failures.append(label)


# ═══════════════════════════════════════════════════════════════════════════════
# A. Source file checks
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}A. Source files{RESET}")

node_file = WORKSPACE_ROOT / "src/perception_pipeline/perception_pipeline/lidar_processor_node.py"
check("lidar_processor_node.py exists", node_file.exists())

setup_text = (WORKSPACE_ROOT / "src/perception_pipeline/setup.py").read_text()
check("lidar_processor entry point in setup.py",
      "lidar_processor = perception_pipeline.lidar_processor_node:main" in setup_text)

launch_text = (WORKSPACE_ROOT / "src/perception_pipeline/launch/fusion_pipeline.launch.py").read_text()
check("lidar_processor in launch file", "lidar_processor" in launch_text)

# ═══════════════════════════════════════════════════════════════════════════════
# B. Import check
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}B. Module imports{RESET}")

try:
    from perception_pipeline.lidar_processor_node import (
        LidarPreprocessor,
        LidarProcessorNode,
        _pc2_to_numpy,
        _numpy_to_pc2,
    )
    check("lidar_processor_node imports OK", True)
except ImportError as e:
    check("lidar_processor_node imports OK", False, str(e))
    print(f"{FAIL}  Cannot continue — aborting.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# C. Preprocessing pipeline unit tests (no ROS needed)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}C. Preprocessing stages{RESET}")

rng = np.random.default_rng(42)
preprocessor = LidarPreprocessor(
    roi_x=(0.0, 50.0), roi_y=(-10.0, 10.0), roi_z=(-3.0, 2.0),
    voxel_size=0.2, ransac_dist=0.2, ransac_iter=100, max_depth=50.0,
)

# Build synthetic cloud: road surface (z≈-1.7) + objects above it
n_ground = 5000
n_objects = 500

# Ground: flat plane at z = -1.7 (roughly correct for KITTI Velodyne height)
ground_pts = np.column_stack([
    rng.uniform(0, 40, n_ground),           # x: forward
    rng.uniform(-8, 8, n_ground),           # y: lateral
    rng.normal(-1.7, 0.05, n_ground),       # z: ground height
    rng.uniform(0, 1, n_ground),            # intensity
]).astype(np.float32)

# Objects: two clusters above ground
obj_a = np.column_stack([
    rng.normal(15, 0.5, n_objects // 2),
    rng.normal(2, 0.3, n_objects // 2),
    rng.uniform(-1.0, 1.0, n_objects // 2),
    np.ones(n_objects // 2) * 0.5,
]).astype(np.float32)

obj_b = np.column_stack([
    rng.normal(30, 0.5, n_objects // 2),
    rng.normal(-3, 0.3, n_objects // 2),
    rng.uniform(-0.8, 1.5, n_objects // 2),
    np.ones(n_objects // 2) * 0.7,
]).astype(np.float32)

# Out-of-ROI points (should be removed)
out_of_roi = np.column_stack([
    rng.uniform(-60, -5, 200),   # behind vehicle (x < 0)
    rng.uniform(-5, 5, 200),
    rng.uniform(-1, 1, 200),
    np.zeros(200),
]).astype(np.float32)

cloud = np.vstack([ground_pts, obj_a, obj_b, out_of_roi])
rng.shuffle(cloud)

result = preprocessor.process(cloud)
s = result["stats"]

# ── ROI crop ──────────────────────────────────────────────────────────────────
check("ROI removes out-of-range points",
      s["n_roi"] < len(cloud),
      f"{len(cloud)} → {s['n_roi']}")
check("ROI count > 0", s["n_roi"] > 0, str(s["n_roi"]))

# ── Voxel downsampling ────────────────────────────────────────────────────────
check("Voxel downsampling reduces count",
      s["n_voxel"] < s["n_roi"],
      f"{s['n_roi']} → {s['n_voxel']}")

# ── Ground removal ────────────────────────────────────────────────────────────
check("Ground points detected",
      s["n_ground"] > 0,
      f"{s['n_ground']} ground points")
check("Ground removal reduces output vs voxel",
      s["n_output"] < s["n_voxel"],
      f"{s['n_voxel']} → {s['n_output']}")

# ── Output shape ──────────────────────────────────────────────────────────────
check("filtered array shape (N, 4)",
      result["filtered"].ndim == 2 and result["filtered"].shape[1] == 4,
      str(result["filtered"].shape))
check("ground array shape (K, 4)",
      result["ground"].ndim == 2 and result["ground"].shape[1] == 4,
      str(result["ground"].shape))
check("filtered dtype float32",
      result["filtered"].dtype == np.float32)

# ── Ground plane geometry check ───────────────────────────────────────────────
if len(result["ground"]) > 0:
    mean_z = result["ground"][:, 2].mean()
    # Ground plane should be near z=-1.7 (we planted it there)
    check("Ground z-mean near planted plane",
          abs(mean_z - (-1.7)) < 0.5,
          f"mean z={mean_z:.3f}")

# ── Non-ground should have object points above ground ─────────────────────────
if len(result["filtered"]) > 0:
    max_z = result["filtered"][:, 2].max()
    check("Non-ground points include above-ground objects",
          max_z > -1.0,
          f"max z={max_z:.3f}")

print(f"\n  Pipeline stats: "
      f"input={s['n_input']}  roi={s['n_roi']}  "
      f"voxel={s['n_voxel']}  ground={s['n_ground']}  output={s['n_output']}")

# ═══════════════════════════════════════════════════════════════════════════════
# D. PointCloud2 conversion round-trip
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}D. PointCloud2 conversion round-trip{RESET}")

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2

pts_orig = rng.random((100, 4)).astype(np.float32)
hdr = Header()
hdr.frame_id = "velodyne"

pc2_msg = _numpy_to_pc2(pts_orig, hdr)
pts_back = _pc2_to_numpy(pc2_msg)

check("Round-trip preserves shape",  pts_back.shape == pts_orig.shape, str(pts_back.shape))
check("Round-trip preserves values", np.allclose(pts_orig, pts_back, atol=1e-6))
check("frame_id preserved",          pc2_msg.header.frame_id == "velodyne")
check("point_step == 16 bytes",      pc2_msg.point_step == 16)

# ═══════════════════════════════════════════════════════════════════════════════
# E. ROS node dry-run
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}E. ROS node dry-run{RESET}")

import rclpy
from rclpy.executors import SingleThreadedExecutor

received = {"filtered": 0, "ground": 0}

rclpy.init()
try:
    processor_node = LidarProcessorNode()
    check("Node instantiated", True)

    # Spy subscriber to count published messages
    from sensor_msgs.msg import PointCloud2 as PC2
    spy_node = rclpy.create_node("spy")
    spy_node.create_subscription(PC2, "/lidar/filtered",     lambda m: received.__setitem__("filtered", received["filtered"]+1), 10)
    spy_node.create_subscription(PC2, "/lidar/ground_plane", lambda m: received.__setitem__("ground",   received["ground"]+1),   10)

    # Publish one synthetic frame to /lidar/points
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                     history=HistoryPolicy.KEEP_LAST, depth=5)
    pub = spy_node.create_publisher(PC2, "/lidar/points", qos)

    pts = np.hstack([ground_pts[:200, :3], np.zeros((200, 1))]).astype(np.float32)
    hdr2 = Header()
    hdr2.frame_id = "velodyne"
    hdr2.stamp = spy_node.get_clock().now().to_msg()
    test_msg = _numpy_to_pc2(pts, hdr2)

    executor = SingleThreadedExecutor()
    executor.add_node(processor_node)
    executor.add_node(spy_node)

    # Publish and spin briefly to process
    deadline = time.monotonic() + 2.0
    published = False
    while time.monotonic() < deadline and rclpy.ok():
        if not published:
            pub.publish(test_msg)
            published = True
        executor.spin_once(timeout_sec=0.05)

    check("Published /lidar/filtered",     received["filtered"] > 0,
          f"{received['filtered']} msgs")
    check("Published /lidar/ground_plane", received["ground"] > 0,
          f"{received['ground']} msgs")

except Exception as e:
    check("Node dry-run", False, str(e))
finally:
    if rclpy.ok():
        rclpy.shutdown()

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 50}")
if failures:
    print(f"\033[91m{BOLD}FAILED{RESET} — {len(failures)} check(s) not passing:")
    for f in failures:
        print(f"  • {f}")
    sys.exit(1)
else:
    print(f"\033[92m{BOLD}ALL CHECKS PASSED{RESET} — Step 3 LiDAR processor is ready.")
print()
