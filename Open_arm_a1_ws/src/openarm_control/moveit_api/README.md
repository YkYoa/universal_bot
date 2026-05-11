# OpenArm MoveIt2 REST API

This package provides a REST API and WebSocket interface to control the OpenArm bimanual robot using MoveIt2.

## 🚀 Quick Start

### 1. Launch the API Stack
This command starts the robot description, MoveIt2 planning nodes, and the REST API server.

```bash
ros2 launch moveit_api robot_api.launch.py use_rviz:=true
```

*   **API Docs**: [http://localhost:5050/api/docs](http://localhost:5050/api/docs)
*   **REST Port**: 5050
*   **WebSocket**: `ws://localhost:5050`

## 🛠 Features

*   **MoveIt2 Integration**: High-level motion planning for `left_arm`, `right_arm`, and `both_arms`.
*   **Advanced Planning Pipelines**:
    *   **OMPL**: Sampling-based planners (RRTConnect, RRTstar, TRRT) with **Ruckig smoothing** for high-quality, jerk-limited motion.
    *   **Pilz Industrial**: Deterministic planners (LIN, PTP) for precise single-arm industrial movements.
*   **REST Endpoints**:
    *   `GET /api/status`: Real-time joint positions and motion state.
    *   `POST /api/move/pose`: Cartesian planning. Now supports `pipeline_id` and `planner_id` selection.
    *   `POST /api/move/joints`: Direct joint-space trajectory control. Supports 14-DOF `both_arms` coordination.
    *   `POST /api/move/named`: Move to predefined poses (e.g., `home`, `ready`) for all groups.
    *   `POST /api/gripper`: Control the left and right grippers.
*   **WebSocket Streaming**: Low-latency joint state updates for UI visualization.

## 📁 Project Structure

*   `launch/robot_api.launch.py`: Main launch file.
*   `moveit_api/robot_api_server.py`: Flask/SocketIO server implementation.
*   `moveit_api/moveit_ee_controller.py`: ROS 2 node bridge to MoveIt2.

## 🔧 Configuration

*   **Port**: Change via environment variable `ROBOT_API_PORT` (default: 5050).
*   **Host**: Change via environment variable `ROBOT_API_HOST` (default: 0.0.0.0).

## 📝 Common Logs & Warnings

The launch file has been optimized to suppress redundant logs. You may still see:
*   `Action server: /recognize_objects not available`: Harmless warning from the RViz Motion Planning display.
*   `No kinematics plugins defined (RViz only)`: A common startup timing warning in RViz; the actual planner in `move_group` is correctly configured.

## 🛑 Clean Shutdown

The system is configured for a clean exit. Use `Ctrl+C` once and wait for the "Shutting down..." logs to complete.


ros2 launch moveit_api robot_api.launch.py \
    use_api:=true \
    use_moveit:=true \
    use_controllers:=true \
    use_rviz:=false


ros2 launch moveit_api robot_api.launch.py \
    use_api:=false \
    use_moveit:=false \
    use_controllers:=false \
    use_rviz:=true \
    use_rsp:=true



