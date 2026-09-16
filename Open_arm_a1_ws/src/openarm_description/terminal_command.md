# Terminal Commands — openarm_description

Package-local commands: build, visualize, and sanity-check the robot
description itself. For full robot bringup/control (real hardware, MoveIt,
REST API), see `Open_arm_a1_ws/terminal_command.md` at the workspace root.

## 0. Build

From `Open_arm_a1_ws/`:
```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select openarm_description --symlink-install
source install/setup.bash
```
`--symlink-install` matters here: it symlinks `urdf`/`config`/`meshes` files
into `install/`, so editing a `.xacro`/`.yaml` and re-launching picks up the
change without a rebuild (a plain `colcon build` copies them instead).

## 1. Visualize (no hardware, no controllers)

**Full robot** — arm(s) + body + base + end-effector, any combination:
```bash
ros2 launch openarm_description display.launch.py \
  ee_type:=amazing_hand body_type:=v2 bimanual:=true mobile_base:=false
```
Key args (all optional, defaults shown):
| arg | default | meaning |
|---|---|---|
| `arm_type` | `v10` | arm hardware version (selects `assets/robot/openarm_v1.0/meshes/arm/<arm_type>/`) |
| `ee_type` | `openarm_hand` | `openarm_hand` or `amazing_hand` |
| `body_type` | `v1` | `v1` (rigid torso) or `v2` (chassis + articulated neck/head) |
| `bimanual` | `false` | two arms instead of one |
| `mobile_base` | `false` | attach the mobile base |
| `xyz_ee` / `rpy_ee` | ee_type-dependent | hand mount offset override |

**Base only:**
```bash
ros2 launch openarm_description display_base.launch.py
```

**amazing_hand only** (slider-driven, no arm):
```bash
ros2 launch openarm_description display_amazing_hand.launch.py \
  side:=right command_space:=knuckle solve_linkage:=true use_gui:=true
```
- `command_space:=knuckle` — sliders drive yaw/flexion; `hand_kinematics_node`
  solves the 8 servo angles + passive linkage (inverse kinematics).
- `command_space:=servo` — sliders drive the 8 SCS0009 horn angles directly
  (forward kinematics for the passive linkage).
- `solve_linkage:=false` — skip `hand_kinematics_node` entirely (e.g. if
  `scipy` isn't installed); passive joints stay at the assembled rest pose.

## 2. amazing_hand under real ros2_control + MoveIt (no arm)

```bash
ros2 launch openarm_description control_amazing_hand.launch.py side:=right use_rviz:=true
```
Starts `controller_manager` + `joint_state_broadcaster` + `j1_group_controller`/
`j2_group_controller` + `move_group` (planning group `hand_fingers`, joint-only —
no IK solver needed). This talks to real ros2_control hardware interfaces;
for a mock/dry-run, use the full-robot `bringup.launch.py` in
`robot_hardware_interface` instead (`use_fake_hardware:=true`).

## 3. Sanity-check a xacro file after editing it

```bash
xacro assets/robot/openarm_v1.0/urdf/v10.urdf.xacro ee_type:=amazing_hand bimanual:=true > /tmp/check.urdf
check_urdf /tmp/check.urdf     # "robot name is: openarm" + no ERROR = valid
```
Do this before committing any change under `assets/` — a broken `xacro:include`
path fails silently as a Command-substitution error at launch time, not at
build time (CMake's `install(DIRECTORY assets ...)` copies files without
parsing them).

## 4. Regenerate the committed `v10.urdf` snapshot

`assets/robot/openarm_v1.0/urdf/v10.urdf` is a checked-in reference snapshot
(not read by any launch file — everything launches from `v10.urdf.xacro`
directly). Regenerate it after changing anything `v10.urdf.xacro` includes:
```bash
xacro assets/robot/openarm_v1.0/urdf/v10.urdf.xacro > assets/robot/openarm_v1.0/urdf/v10.urdf
```

## 5. Regenerate Doxygen docs (workspace-wide)

```bash
cd Open_arm_a1_ws && doxygen Doxyfile
```
Output: `Open_arm_a1_ws/docs/doxygen/html/index.html`. Covers this package's
`.py` files (launch scripts, `hand_kinematics_node.py`); `.xacro`/`.urdf`
files are XML, not part of Doxygen's C/Python parsers, so they're documented
with structured `<!-- -->` header comments instead (see any file under
`assets/` for the convention).
