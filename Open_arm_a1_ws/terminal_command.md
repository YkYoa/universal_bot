# Terminal Commands — OpenArm A1 (IQ9075)

Full pipeline: connect → check → calibrate/zero → bring up → MoveIt → control.
Device: `ssh ubuntu@192.168.1.226` (see project memory `reference-device-iq9075`).

## 0. Connect + environment

```bash
ssh ubuntu@192.168.1.226
```

Only needed for **non-interactive** one-liners (`ssh host "cmd"`) — `.bashrc` isn't sourced then:
```bash
export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///home/ubuntu/.ros/cyclonedds.xml
source /opt/ros/jazzy/setup.bash
source ~/arm_ws/install/setup.bash
```
If logged in interactively, these are already set — just check nothing stale is running before bringup:
```bash
ps aux | grep ros2_control_node   # should show nothing but the grep itself
```
Two control loops fighting over the same CAN bus is a real hazard — `Ctrl+C` the old bringup terminal or `pkill -f ros2_control_node` first.

## 1. CAN — check

`left_can: can1`, `right_can: can0` (per `robot_hardware_interface/config/hardware_config.yaml`).

```bash
ip -details -statistics link show can0
ip -details -statistics link show can1
candump can0        # raw frames, Ctrl+C to stop
```
If down and you need to bring one up manually (bringup.launch.py does this automatically if the adapter is live):
```bash
sudo openarm-can-configure-socketcan-4-arms     # or manually:
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up
```

## 2. CAN — calibration / zero

Full runbook: `~/arm_ws/src/control/robot_hardware_interface/CALIBRATION.md`. Use when a joint's reported position no longer matches its real physical position.

**Stop bringup first** (see step 0) — the calibration tool needs the CAN bus to itself.

**⚠️ Never insert a `sleep`/delay between `disable` and `set_zero`.** Once torque is off, unbraked joints (esp. wrist joint5) sag under gravity within 1-2s and corrupt the zero. Run the two back-to-back.

```bash
# 1. Disable torque (arm goes limp — support it by hand first)
openarm-can-cli -i <can_interface> disable --id <motor_id(s)>

# 2. By hand, move joint(s) to the position that should become zero
#    (arm joints 1-7 → true mechanical home/straight pose)

# 3. Record it — run immediately after step 2, no delay
openarm-can-cli -i <can_interface> set_zero --id <motor_id(s)>

# 4. Verify BEFORE re-enabling — Motor/Output Shaft Position should read ~0.000
openarm-can-cli -i <can_interface> show_param --id <motor_id(s)>

# 5. Re-enable torque
openarm-can-cli -i <can_interface> enable --id <motor_id(s)>
```

**Left arm joints 1-7** (`can1`):
```bash
openarm-can-cli -i can1 disable --id 1,2,3,4,5,6,7
openarm-can-cli -i can1 set_zero --id 1,2,3,4,5,6,7
openarm-can-cli -i can1 show_param --id 1,2,3,4,5,6,7
openarm-can-cli -i can1 enable --id 1,2,3,4,5,6,7
```
**Right arm joints 1-7** (`can0`): same commands, `-i can0`.

**Motor 8 (gripper/amazing_hand connector)** — same `disable`/`set_zero`/`show_param`/`enable` pattern with `--id 8` on the appropriate interface. **Do NOT** use the vendor's `openarm-can-zero-position-calibration` tool on motor 8 if `ee_type=amazing_hand` — it seeks a mechanical hard stop that doesn't exist in that direction and will spin indefinitely (has happened before). That tool is still fine for joints 1-7.

After recalibrating motor 8 by a non-trivial amount, re-check `hand_rotate_lower`/`hand_rotate_upper` in `openarm_description/assets/end_effector/amazing_hand/urdf/amazing_hand_arguments.xacro` still make physical sense.

## 3. Bring up

Three different launch entry points depending on what you need:

**A. Dev/test bringup** (arms + MoveIt move_group + optional RViz, self-contained config):
```bash
ros2 launch robot_hardware_interface bringup.launch.py arms:=auto head:=auto use_rviz:=true ee_type:=amazing_hand body_type:=v2
```
`arms`/`head`: `auto` (default, detects live CAN/ethernet link), or force `true`/`false`.
--------------------------------------------------------------
control robot arm through topic list 


**A2. One arm real, the other fake** (added 2026-08-28 — `arms:=` alone is all-or-nothing and previously crashed `ros2_control_node` entirely if the other side's CAN wasn't ready). Use `left_arm:=`/`right_arm:=` to override a side independently — `same` (default) follows `arms:=`, or force `true`/`false`/`auto` per side:
```bash
ros2 launch robot_hardware_interface bringup.launch.py left_arm:=true right_arm:=false head:=false use_rviz:=false ee_type:=openarm_hand body_type:=v2 2>&1 | tee ~/bringup_left_test.log
```
Check afterward: `ros2 control list_hardware_components` — left should show `active`/real, right `mock_components/GenericSystem`/fake, no crash of the other side.

**B. Full REST/WebSocket control API** (MoveIt + REST API on port 5050 + dashboard):
```bash
ros2 launch moveit_api robot_api.launch.py use_rviz:=false use_moveit:=true use_api:=true use_controllers:=true ee_type:=amazing_hand body_type:=v2
```

**C. Production app launch** (qvic_2026 — sequences, DB, API bundled):
```bash
ros2 launch qvic_2026 qvic_2026.launch.py arm:=both ee_type:=none head:=true use_fake_hardware:=false use_rviz:=false use_api:=true use_db:=true autostart:=false
```
Set `use_fake_hardware:=true` for a dry run without touching real motors.

**C2. Local dry-run on the laptop, isolated from IQ9075** (added 2026-09-11). The laptop's default `ROS_DOMAIN_ID=42` + `~/.ros/cyclonedds.xml` (see project memory `dds-rmw-setup-iq9075`) unicast-peer with `192.168.1.226` by design — fine for remote RViz, but it means a plain local launch shares the same DDS domain as a live IQ9075 bringup: duplicate node names (`robot_state_publisher`, `controller_manager`, `move_group`, ...) and topics (`/joint_states`, controller commands, ...) can crosstalk with the real robot. Override both per-invocation (don't touch `.bashrc`'s defaults, which you want when actually bridging to IQ9075):
```bash
ROS_DOMAIN_ID=99 CYCLONEDDS_URI="" ros2 launch qvic_2026 qvic_2026.launch.py \
  arm:=both ee_type:=amazing_hand head:=true use_fake_hardware:=true use_rviz:=true \
  use_api:=false use_db:=true autostart:=false
```
- Use `export` (not a bare `VAR=val` line) if setting these in a separate terminal from the one launching — a bare assignment only sticks for that shell if the var was already exported elsewhere (e.g. `ROS_DOMAIN_ID` via `.bashrc`); it silently won't reach a fresh, unrelated terminal's `ros2` calls otherwise.
- `ee_type` must match what the sequence data was recorded for — `qvic_2026`'s `sequence.yaml` (`homePoses`'s `lhHomeYaw`/`lhHomeFlex`/etc., and the extra 8th "motor 8" value in `laHomeAngle`/`raHomeAngle` — `both_arms` is 16-DOF under `amazing_hand`, 14 under `openarm_hand`/`none`) is `amazing_hand` data (`left_hand_j1_controller` etc.); `openarm_hand`/`none` will fail hand-pose steps with `.../follow_joint_trajectory not available` even though arm-only steps succeed either way.
- If a controller_manager overrun WARN (`Overrun detected! ... missed cycles`) causes a step to fail with `Blocking execution failed with status: TIMED_OUT`, it's `ros2_control_node` not getting real-time scheduling (`Could not enable FIFO RT scheduling policy: Operation not permitted` at startup). Fix once, needs a relogin/reboot to take effect:
  ```bash
  echo "hans   -  rtprio  98" | sudo tee /etc/security/limits.d/99-ros2-control-rtprio.conf
  # log out/in or reboot, then confirm: ulimit -Hr  →  should print 98, not 0
  ```

### Local run with the REST API + web dashboard (`use_api:=true`)

`robot_api_server` needs Flask — install via **apt**, not pip (`pip install --user` hits `externally-managed-environment` on this box, and a venv would need `--system-site-packages` to still see the apt-installed `rclpy`; apt is simplest and matches this package's own `package.xml` deps):
```bash
sudo apt install python3-flask python3-flask-socketio python3-flask-compress
```
Then launch with `use_api:=true`:
```bash
ROS_DOMAIN_ID=99 CYCLONEDDS_URI="" ros2 launch qvic_2026 qvic_2026.launch.py \
  arm:=both ee_type:=amazing_hand head:=true use_fake_hardware:=true use_rviz:=false \
  use_api:=true use_db:=true autostart:=false
```
Open in a browser:
- `http://localhost:5050/dashboard/fsm.html` — sequence picker + **Run**/**Dry run** buttons, plus Pause/Resume/Step/Cancel/**Clear fault**/E-stop. This is the fast path for repeated testing — no more retyping `ros2 action send_goal`/`service call fsm_command` by hand each time.
- `http://localhost:5050/dashboard/` — 3D pose/joint-state visualizer.
- `http://localhost:5050/api/docs` — full REST API reference.

CLI equivalents (no API/dashboard needed — node name is always `sequence_executor_node` regardless of package):
```bash
ros2 action send_goal /sequence_executor_node/run_sequence openarm_messages/action/RunSequence \
  "{sequence_name: 'qvic_2026_both', repeat_override: 0, velocity_override: 0.0, dry_run: false}" --feedback
ros2 service call /sequence_executor_node/fsm_command openarm_messages/srv/FsmCommand "{command: 'clear_fault'}"
```

## 4. MoveIt — remote RViz from laptop

Full details + gotchas in project memory `dds-rmw-setup-iq9075`. Quick version — laptop needs matching `ROS_DOMAIN_ID=42`/`RMW_IMPLEMENTATION`/`CYCLONEDDS_URI` and CycloneDDS peers/interface pinned, then:
```bash
rviz2 -d ~/.ros/iq9075_bringup.rviz --ros-args \
  --params-file ~/.ros/iq9075_semantic_params.yaml \
  --params-file ~/.ros/iq9075_kinematics.yaml \
  --params-file ~/.ros/iq9075_pipeline_params.yaml
```

## 5. Control

**Via REST API** (bringup option B or C with `use_api:=true`, port 5050) — see `moveit_api/README.md` for full docs, live at `GET http://192.168.1.226:5050/api/docs`.

Planning groups: `left_arm`, `right_arm`, `both_arms`, `left_hand_fingers`, `right_hand_fingers`, `head`.

```bash
# Move every joint of a group (degrees here, radians is the default)
curl -X POST http://192.168.1.226:5050/api/move/joints \
  -H "Content-Type: application/json" \
  -d '{"group":"left_arm","positions":[0,-10,0,45,0,30,0],"unit":"deg","velocity_scaling":0.3}'

# Move a single joint, everything else holds position
curl -X POST http://192.168.1.226:5050/api/move/joint \
  -H "Content-Type: application/json" \
  -d '{"group":"left_arm","joint":"openarm_left_joint4","value":45,"unit":"deg"}'

# Named pose
curl -X POST http://192.168.1.226:5050/api/move/named \
  -H "Content-Type: application/json" \
  -d '{"group":"left_hand_fingers","pose":"open"}'

# Cartesian pose (arms only)
curl -X POST http://192.168.1.226:5050/api/move/pose \
  -H "Content-Type: application/json" \
  -d '{"group":"left_arm","position":{"x":0.4,"y":0.2,"z":1.1},"orientation":{"x":0,"y":0,"z":0,"w":1}}'

# Status / feedback
curl http://192.168.1.226:5050/api/status
curl http://192.168.1.226:5050/api/pose/left_arm
```
3D dashboard (no RViz needed): `http://192.168.1.226:5050/dashboard/`.

**Via ros2_control directly** (no REST API running):
```bash
ros2 control list_controllers
ros2 topic pub /left_arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "..."
```

## 6. Status / health checks

```bash
ros2 control list_controllers
ros2 control list_hardware_components
ros2 control list_hardware_interfaces
ros2 topic echo /joint_states --once
ros2 topic hz /joint_states           # expect ~100Hz
ros2 topic echo /diagnostics --once   # if published
```
Healthy bimanual bringup = 4 active controllers (`head_controller`, `left_arm_controller`, `right_arm_controller`, `joint_state_broadcaster`) and 4 active hardware components (head/left/right real, base is normally `mock_components/GenericSystem` — no real base hardware yet).

Logs already on-device (`~/`): `monitor_left_*.log`, `monitor_right_*.log`, `joint_states_*watch*.log`, `qvic_restart.log`, `head_api.log`, `hand_api.log`, `gateway_api.log`.

System health:
```bash
cat /sys/class/thermal/thermal_zone*/temp   # milli-°C, divide by 1000
top -bn1 | head -20
free -h
df -h /
```

## 7. Deploy code changes to the robot

From laptop, inside `Open_arm_a1_ws/`:
```bash
./deploy_to_robot.sh
```
Rsyncs `src/` (excludes build/install/log) to the robot's `~/arm_ws/src/` and runs `colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release` there. Targets `192.168.1.226` (LAN, matches this doc's SSH host).

## 8. Known hardware gotchas

- **Left arm joint1 mechanical limit ≈ -90° to +30°** (`-1.5708`..`+0.5236` rad) — real mechanism limit, tighter than the URDF/software range. Don't plan/command `openarm_left_joint1` outside this without confirming the current physical limit first.
- All 8 motors per arm confirmed physically connected as of 2026-08-17 (an earlier "motor 8 unplugged" note is stale).
- CAN mapping: `left_can → can1`, `right_can → can0`.

## 9. moveit_servo — real-time jogging

One `servo_node` per arm (`control/openarm_servo` package), namespaced `left_arm_servo`/`right_arm_servo` — separate from the offline OMPL/Pilz planning pipelines in `openarm_moveit_config`. `servo_params.yaml`'s velocity numbers are reasoned from real `joint_limits.yaml`/`planner_profiles.yaml` data, but singularity thresholds and collision proximity margins are still moveit_servo's generic defaults — not yet validated against the real arm. **Test on fake hardware first.**

```bash
# 0. One-time: install moveit_servo, then rebuild the two new packages
sudo apt install ros-jazzy-moveit-servo
cd /home/hans/universal_bot/Open_arm_a1_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select openarm_servo openarm_moveit_config --symlink-install
source install/setup.bash

# 1. Bringup on fake hardware + RViz (separate terminal)
ros2 launch openarm_moveit_config moveit_bimanual.launch.py use_fake_hardware:=true use_rviz:=true

# 2. Servo nodes (separate terminal, source install/setup.bash again)
ros2 launch openarm_servo servo.launch.py use_fake_hardware:=true

# 3. Interactive keyboard jog test (separate terminal) - left arm example
ros2 run moveit_servo servo_keyboard_input --ros-args -r cmd_vel:=/left_arm_servo/delta_twist_cmds
```
Watch RViz for smooth motion and the servo terminal for `DECELERATE_FOR_SINGULARITY`/`HALT_FOR_COLLISION`/`JOINT_BOUND` warnings before ever pointing this at the real robot (`use_fake_hardware:=false`, same launch commands, run on IQ9075 not the laptop).

Programmatic jogging (C++, e.g. from a future teleop node): `openarm_servo::ServoClient` (`include/openarm_servo/servo_client.hpp`) wraps the raw topics — `jogCartesian()`, `jogJoint()`, `stop()`, `setStatusCallback()`. It has zero awareness of the FSM/`motors_enabled_` - a caller that must not jog during FAULT/E-stop has to check `~/state` itself first.
