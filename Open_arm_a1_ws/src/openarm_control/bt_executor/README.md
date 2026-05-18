# bt_executor — OpenArm VLA Behavior Tree Engine

## Overview

This package replaces the direct REST → MoveIt call chain with a
**BehaviorTree.CPP v4** tick loop sitting between the VLA layer and the
motion stack.

```
VLA bridge  →  Blackboard  ←→  BT engine (50 Hz)  →  MoveIt2 / ros2_control
                                     ↓
                              REST API (UI telemetry only)
```

The VLA model **writes intent** (poses, object labels, confidence) to the
shared blackboard.  The BT **decides when and whether to act** — including
all precondition checks and recovery fallbacks.

---

## Package structure

```
bt_executor/
├── include/bt_executor/
│   ├── blackboard_keys.hpp          ← all BB key constants (start here)
│   └── nodes/
│       ├── conditions/
│       │   ├── is_object_visible.hpp
│       │   ├── is_gripper_holding.hpp
│       │   ├── is_arm_at_pose.hpp
│       │   └── is_replan_needed.hpp
│       └── actions/
│           ├── move_to_named_pose.hpp
│           ├── move_to_pose.hpp
│           ├── grasp_object.hpp      (+ release_object in same file)
│           ├── query_vla.hpp
│           └── safe_abort.hpp
├── src/
│   ├── bt_executor_node.cpp         ← main(), factory registration, tick loop
│   └── nodes/
│       ├── conditions/is_object_visible.cpp
│       └── actions/move_to_pose.cpp, move_to_named_pose.cpp
├── bt_trees/
│   └── pick_and_place.xml           ← runtime-loadable tree (edit with Groot2)
├── launch/
│   └── bt_executor.launch.py
└── config/                          ← add node-specific params here
```

---

## Build

```bash
cd Open_arm_a1_ws

# Install BehaviorTree.CPP v4 and behaviortree_ros2
sudo apt install ros-humble-behaviortree-cpp
# behaviortree_ros2 may need to be built from source:
# https://github.com/BehaviorTree/BehaviorTree.ROS2

colcon build --packages-select bt_executor --symlink-install
source install/setup.bash
```

---

## Run

```bash
# Full stack (fake hardware + MoveIt + BT engine)
ros2 launch bt_executor bt_executor.launch.py

# With RViz and the REST API for UI telemetry
ros2 launch bt_executor bt_executor.launch.py use_rviz:=true use_api:=true

# Custom tree
ros2 launch bt_executor bt_executor.launch.py \
  bt_xml_path:=/path/to/my_tree.xml
```

---

## Monitor & debug

```bash
# Watch BT tick status
ros2 topic echo /bt_executor/status

# Watch fault alerts
ros2 topic echo /bt_executor/fault

# Manually trigger replan (simulates VLA confidence drop)
ros2 service call /bt_executor/replan std_srvs/srv/Trigger

# Inject a grasp pose for testing (bypasses VLA)
ros2 topic pub /bt_test/grasp_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: world}, pose: {position: {x: -0.3, y: 0.15, z: 0.8}, \
   orientation: {w: 1.0}}}" --once
```

Visualize the live tree with **Groot2**:
```bash
groot2
# Connect to: localhost:1667  (FileLogger port, enable log_to_file:=true)
```

---

## TODO map (search for "TODO:" in source)

### High priority — needed before real hardware

| File | TODO |
|---|---|
| `bt_executor_node.cpp` | Wire `StateMonitor` blackboard pointer after `initialize()` |
| `bt_executor_node.cpp` | Replace `nullptr` StateMonitor with real blackboard ref |
| `is_object_visible.cpp` | Replace stub with real world-model map lookup |
| `is_gripper_holding.hpp` | Wire force sensor topics → BB_LEFT/RIGHT_GRIP_FORCE |
| `safe_abort.hpp` | Implement controller halt (cancel active FJT goals) |
| `safe_abort.hpp` | Implement gripper open on abort |
| `query_vla.hpp` | Promote to `RosActionNode<openarm_msgs::action::QueryVLA>` |
| `query_vla.hpp` | Poll `BB_NEW_GOAL_READY` from real VLA bridge instead of stub |

### Medium priority

| File | TODO |
|---|---|
| `grasp_object.hpp` | Attach object to EE in MoveIt planning scene (`/apply_planning_scene`) |
| `release_object.hpp` | Detach object from planning scene on release |
| `move_to_pose.cpp` | Add Ruckig smoothing pipeline option (already in ompl_planning.yaml) |
| `bt_executor_node.cpp` | Subscribe to `/bt_executor/new_goal` topic for VLA bridge to push tasks |

### Nice to have

| Item | Notes |
|---|---|
| `BimanualPickAndPlace.xml` | Extend tree to coordinate both arms via Parallel node |
| Groot2 monitor integration | Enable `ZmqPublisher` logger for live tree visualization |
| `openarm_msgs` package | Define `QueryVLA.action` and `WorldModel.msg` |

---

## Adding a new BT node

1. Create `include/bt_executor/nodes/actions/my_node.hpp`
2. Inherit from `BT::RosActionNode<MyActionType>` (async) or `BT::SyncActionNode`
3. Implement `providedPorts()`, `setGoal()`, `onResultReceived()`, `onFailure()`
4. Register in `bt_executor_node.cpp`:
   ```cpp
   BT::RegisterRosAction<MyNode>(factory_, "MyNode", ros_params);
   ```
5. Add the XML tag to your tree file and reload — no recompile needed for
   tree structure changes.
