"""
fusion_pipeline.launch.py
=========================
Launches all nodes in the LiDAR-Camera fusion pipeline.

Environment variables (override defaults):
  KITTI_SEQ   Path to the KITTI raw sync sequence directory (required)
  FRAME_RATE  Playback rate in Hz (default: 10.0)
  LOOP        Loop the sequence: true/false (default: true)

Example:
  KITTI_SEQ=/data/kitti/2011_09_26/2011_09_26_drive_0001_sync pixi run launch
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Arguments (can be overridden on CLI or via env vars) ─────────────────
    seq_arg = DeclareLaunchArgument(
        "sequence_path",
        default_value=os.environ.get("KITTI_SEQ", ""),
        description="Path to KITTI raw sync sequence directory",
    )
    rate_arg = DeclareLaunchArgument(
        "frame_rate",
        default_value=os.environ.get("FRAME_RATE", "10.0"),
        description="Playback rate in Hz",
    )
    loop_arg = DeclareLaunchArgument(
        "loop",
        default_value=os.environ.get("LOOP", "true"),
        description="Loop the sequence when it ends",
    )

    # ── KITTI publisher (Step 2) ───────────────────────────────────────────────
    kitti_publisher = Node(
        package="perception_pipeline",
        executable="kitti_publisher",
        name="kitti_publisher",
        parameters=[{
            "sequence_path": LaunchConfiguration("sequence_path"),
            "frame_rate": LaunchConfiguration("frame_rate"),
            "loop": LaunchConfiguration("loop"),
        }],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        seq_arg,
        rate_arg,
        loop_arg,
        LogInfo(msg="Starting LiDAR-Camera Fusion Pipeline (KITTI playback)"),
        kitti_publisher,
        # camera_detector and lidar_processor nodes added in later steps
    ])
