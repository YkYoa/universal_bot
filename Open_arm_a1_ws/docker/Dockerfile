# ─────────────────────────────────────────────────────────────────────────────
# OpenArm VLA Bridge — ROS 2 Humble container
# Chỉ build openarm_messages (action types) + cài vla_bridge.py Python script
# KHÔNG build C++ packages (planning_interface, bt_executor C++, motion_planner)
# ─────────────────────────────────────────────────────────────────────────────
FROM ros:humble-ros-base-jammy

LABEL maintainer="OpenArm Team"
LABEL description="VLA Bridge Python node for OpenArm — ROS 2 Humble"

# ── 1. System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-cv-bridge \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-action-msgs \
    ros-humble-rclpy \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-default-runtime \
    ros-humble-rosidl-default-generators \
    python3-opencv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Python dependencies ────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    requests \
    "numpy<2"

# ── 3. Copy workspace source ──────────────────────────────────────────────────
WORKDIR /openarm_ws
COPY src/ src/

# ── 4. Build ONLY openarm_messages (action/msg definitions) ──────────────────
# vla_bridge.py là Python script — không cần compile C++ gì cả
# Chỉ cần build message definitions để generate Python bindings
RUN apt-get update && \
    . /opt/ros/humble/setup.sh && \
    rosdep update --rosdistro humble && \
    rosdep install --from-paths src/communication/messages \
      --ignore-src -r -y --rosdistro humble && \
    rm -rf /var/lib/apt/lists/*

RUN . /opt/ros/humble/setup.sh && \
    colcon build \
      --packages-select openarm_messages \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build/ log/

# ── 5. Install vla_bridge.py Python script ────────────────────────────────────
# Copy script vào thư mục lib của ROS 2 để có thể chạy bằng `ros2 run bt_executor vla_bridge.py`
RUN mkdir -p /openarm_ws/install/bt_executor/lib/bt_executor && \
    cp /openarm_ws/src/openarm_control/bt_executor/scripts/vla_bridge.py \
       /openarm_ws/install/bt_executor/lib/bt_executor/vla_bridge.py && \
    chmod +x /openarm_ws/install/bt_executor/lib/bt_executor/vla_bridge.py

# Tạo package marker để ros2 biết package bt_executor tồn tại
RUN mkdir -p /openarm_ws/install/bt_executor/share/bt_executor && \
    echo "bt_executor" > /openarm_ws/install/bt_executor/share/bt_executor/package.dsv

# ── 6. CycloneDDS config ──────────────────────────────────────────────────────
COPY docker/cyclonedds_server.xml /openarm_ws/cyclonedds.xml
ENV CYCLONEDDS_URI=file:///openarm_ws/cyclonedds.xml
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV ROS_DOMAIN_ID=42

# ── 7. Entrypoint ─────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
