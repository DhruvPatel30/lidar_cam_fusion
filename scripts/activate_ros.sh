#!/usr/bin/env bash
# Sourced automatically by pixi on environment activation.
# Sets up ROS 2 Humble environment variables.

if [ -f "$CONDA_PREFIX/setup.bash" ]; then
    source "$CONDA_PREFIX/setup.bash"
elif [ -f "$CONDA_PREFIX/share/ros2_humble/setup.bash" ]; then
    source "$CONDA_PREFIX/share/ros2_humble/setup.bash"
fi

# Add local workspace install to ROS path (after colcon build)
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$WORKSPACE_DIR/install/setup.bash" ]; then
    source "$WORKSPACE_DIR/install/setup.bash"
fi

export ROS_DOMAIN_ID=0
export RCUTILS_COLORIZED_OUTPUT=1
