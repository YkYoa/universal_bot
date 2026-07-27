# OpenArm MoveIt2 REST API

REST + WebSocket interface to control the OpenArm bimanual robot (arms, amazing_hand fingers, v2 body head) via MoveIt2. All motion is collision-checked (MoveGroup) — a request that would hit something, or that MoveIt can't plan, returns `success: false` with a reason instead of executing.

Sequence choreography (multi-step demo routines) is being moved to the `openarm_head_hand` ROS package instead of this API — use this API for direct / ad-hoc control (single moves, testing, live control from the app).

## 🚀 Quick Start

```bash
ros2 launch moveit_api robot_api.launch.py use_rviz:=true
```

* **API Docs (live, matches running code)**: `GET http://<robot-ip>:5050/api/docs`
* **REST port**: 5050
* **WebSocket**: `ws://<robot-ip>:5050`

All requests/responses are JSON. All endpoints return `{"success": true/false, "message": "..."}` at minimum; `success: false` means either the group/joint was invalid (HTTP 400) or MoveIt failed to plan/execute — e.g. collision, unreachable, no IK solution (HTTP 422). Always check `success`, don't assume HTTP 200 alone means the robot moved.

## Planning groups

Every group-taking endpoint accepts any of these — pick the one you want to move:

| Group | Joints | Order | Notes |
|---|---|---|---|
| `left_arm` | 7 | `openarm_left_joint1..7` | |
| `right_arm` | 7 | `openarm_right_joint1..7` | |
| `both_arms` | 14 | left 7 then right 7 | moves both arms in one synced plan |
| `left_hand_fingers` | 8 | `j11,j12,j13,j14, j21,j22,j23,j24` | yaw then flex, per finger 1-4. Thumb = finger 4 (`j14`/`j24`) |
| `right_hand_fingers` | 8 | same order, right side | |
| `head` | 2 | `neck_joint` (pan), `head_joint` (tilt) | no named poses yet — joint control only |

`GET /api/docs` returns this same table live (`planning_groups`), so it always matches whatever's actually deployed.

## Units: radians or degrees

Every joint value defaults to **radians**. Pass `"unit": "deg"` to use degrees instead — applies to `/api/move/joints` and `/api/move/joint`.

## Controlling joints

### Move every joint of a group at once

```
POST /api/move/joints
{
  "group": "left_arm",
  "positions": [0, -10, 0, 45, 0, 30, 0],
  "unit": "deg",
  "velocity_scaling": 0.3
}
```
`positions` must have exactly as many values as the group has joints (see table above), in that group's joint order.

### Move exactly one joint (holds everything else where it is)

```
POST /api/move/joint
{
  "group": "left_arm",
  "joint": "openarm_left_joint4",
  "value": 45,
  "unit": "deg"
}
```
`joint` can be the joint name, or its 0-based index within the group (e.g. `3` for the 4th joint). Every other joint in the group stays at its current measured position — this is the one to use for a per-joint slider in the app.

Head example (tilt the head down 20°, leave pan alone):
```
POST /api/move/joint
{"group": "head", "joint": 1, "value": -20, "unit": "deg"}
```

### Move to a Cartesian pose

```
POST /api/move/pose
{"group": "left_arm", "position": {"x":0.4,"y":0.2,"z":1.1}, "orientation": {"x":0,"y":0,"z":0,"w":1}}
```
Arms only (`left_arm`/`right_arm`). Not applicable to hands/head.

### Named poses

```
POST /api/move/named
{"group": "left_hand_fingers", "pose": "open"}
```
Valid poses: arms → `home`/`ready`. Hands → `open`/`close`/`home`. Head has none yet (use joint control).

### Gripper (2-finger `openarm_hand` end effector only, not amazing_hand)

```
POST /api/gripper
{"side": "left", "action": "open"}
```

### amazing_hand shortcut (5-finger hand — equivalent to `/api/move/named` or `/api/move/joints` on the hand groups, kept for convenience)

```
POST /api/hand
{"side": "left", "action": "close"}
POST /api/hand
{"side": "left", "positions": [0,0,0,0, 1,1,1,1]}
```

## Status / feedback

```
GET /api/status         # all current joint positions + is_moving flags
GET /api/pose/<group>   # current EE pose via FK, arms only
```
WebSocket `subscribe_joint_states` streams `/joint_states` at 10Hz if you want live feedback instead of polling.

## 🔧 Configuration

* **Port**: `ROBOT_API_PORT` env var (default 5050)
* **Host**: `ROBOT_API_HOST` env var (default 0.0.0.0)

## 🛑 Running it

```bash
ros2 launch moveit_api robot_api.launch.py \
    use_api:=true use_moveit:=true use_controllers:=true use_rviz:=false
```
`use_rviz:=true` opens RViz too (needs `DISPLAY`/`XAUTHORITY` set if running headless over SSH into a desktop session). `Ctrl+C` once and wait for the shutdown logs — don't kill -9 it, that leaves orphaned `move_group`/`ros2_control_node` processes that block the next launch.
