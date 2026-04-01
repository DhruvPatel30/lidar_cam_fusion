#!/usr/bin/env bash
# Distro-agnostic ROS 2 activation — works for Jazzy and any future distro.

# 1. Source the ROS underlay that robostack installs.
if [[ -f "$CONDA_PREFIX/setup.bash" ]]; then
    source "$CONDA_PREFIX/setup.bash"
else
    # Fallback: scan for any ros2_<distro>/setup.bash
    for f in "$CONDA_PREFIX"/share/ros2_*/setup.bash; do
        [[ -f "$f" ]] && source "$f" && break
    done
fi

# 2. Source colcon workspace overlay (present after: pixi run build).
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$WORKSPACE_DIR/install/setup.bash" ]]; then
    source "$WORKSPACE_DIR/install/setup.bash"
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RCUTILS_COLORIZED_OUTPUT=1
