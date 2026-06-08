#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Docker entrypoint — source ROS và workspace, rồi chạy lệnh truyền vào
# ─────────────────────────────────────────────────────────────────────────────
set -e

# Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# Source built workspace (openarm_messages + bt_executor install)
if [ -f /openarm_ws/install/setup.bash ]; then
    source /openarm_ws/install/setup.bash
fi

# Thêm bt_executor script vào PATH để `ros2 run` tìm thấy
export PATH="/openarm_ws/install/bt_executor/lib/bt_executor:$PATH"

# Ensure env vars
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///openarm_ws/cyclonedds.xml
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}

echo "[entrypoint] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[entrypoint] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "[entrypoint] CYCLONEDDS_URI=${CYCLONEDDS_URI}"
echo "[entrypoint] Ready."

exec "$@"
