# qvic_2026

Project home for the QVIC 2026 (Qualcomm) competition entry. Holds this
project's own `sequence.yaml`, generated BT XML files, and launch setup.
Reuses generic tooling from `control/` and `builds/openarm_demo/` rather
than duplicating it — this package should stay thin.

## Directory layout

```
qvic_2026/
├── config/
│   ├── sequence.yaml           # this project's waypoints/poses (source of truth)
│   └── qvic_2026*_bt.xml       # generated BT trees (regenerate from sequence.yaml, don't hand-edit)
├── launch/
│   └── qvic_2026.launch.py     # launches bt_executor_node against a given bt_xml_path
├── src/                        # project-specific C++ (empty so far - see "Adding project code" below)
├── include/
├── CMakeLists.txt
└── package.xml
```

## The pipeline, end to end

Every sequence this project runs goes through the same four stages:

**1. Get waypoints into `config/sequence.yaml`** — two ways:

- **Hand-guide + record** (`control/waypoint_recorder`): put the arm in
  `control_mode: torque` (gravity compensation active), physically move it,
  and record positions live.
  ```
  ros2 run waypoint_recorder record_waypoint --loop --auto-convert --file /home/hans/universal_bot/Open_arm_a1_ws/src/builds/qvic_2026/config/sequence.yaml --bt-out /home/hans/universal_bot/Open_arm_a1_ws/src/builds/qvic_2026/config/qvic_2026_bt.xml
  ```
  Interactive: pick arm (`la`/`ra`/`both`), pick/create a section, then
  press Enter per waypoint (auto-numbered `<arm><Section><N>Angle`), `s` to
  switch section, `q` to finish (auto-converts to BT XML on exit).

- **Formula-generated** (`control/trajectory_waypoint_generator`): compute
  a shape (line, ellipse arc, circle, square) between reference poses
  instead of hand-recording every point.
  ```
  ros2 run trajectory_waypoint_generator generate_waypoint_sequence --arm la --section waveEllipse --shape ellipse --start key:laPreWaveJoint --end key:laEndWaveJoint --file /home/hans/universal_bot/Open_arm_a1_ws/src/builds/qvic_2026/config/sequence.yaml --num-points 20 --height-ratio 0.3
  ```
  `--start`/`--end` (or `--center` for circle/square) each accept `live`
  (capture the arm's current position now) or `key:NAME` (read an existing
  waypoint out of `--file`, no live robot needed). See
  `control/trajectory_waypoint_generator/src/generate_waypoint_sequence.cpp`'s
  header comment and `--help` for full shape-specific options.

Both tools write the same key format:
`<arm_prefix><PascalCase(section)><N>Angle` (e.g. `laWaveEllipse1Angle`).
**`--section` must not end in a digit** — the trailing section digit and
the point-number digits would concatenate ambiguously.

**2. Convert a section to BT XML** (skip this if you used `--auto-convert`
above):
```
ros2 run openarm_demo sequence_to_bt --yaml /home/hans/universal_bot/Open_arm_a1_ws/src/builds/qvic_2026/config/sequence.yaml --out /home/hans/universal_bot/Open_arm_a1_ws/src/builds/qvic_2026/config/qvic_2026_<name>_bt.xml --section <sectionName> --arm left_arm
```
A run of consecutive auto-numbered keys in one section is automatically
merged into a single `PlanToJointSequence` BT node (one continuous
blended trajectory) instead of N separate `PlanToJointTarget` nodes (N
independent stop-start moves) — this happens automatically based on the
`<N>` numbering, no extra flag needed.

**3. Rebuild** so the installed `share/` copy picks up the new files:
```
colcon build --packages-select qvic_2026
```
(run from `~/universal_bot/Open_arm_a1_ws`)

**4. Replay — fake hardware first, always:**
```
ros2 launch bt_executor bt_executor.launch.py bt_xml_path:=/home/hans/universal_bot/Open_arm_a1_ws/install/qvic_2026/share/qvic_2026/config/qvic_2026_<name>_bt.xml use_rviz:=true
```
This is a fully self-contained fake-hardware stack (robot_state_publisher,
fake ros2_control, move_group, robot_skills_node, bt_executor, RViz) — one
command, no other terminals needed. Confirm the motion looks right in RViz
*before* touching real hardware.

**Real hardware** (only after the fake-hardware dry run looks right), 3
terminals:
```
ros2 launch robot_hardware_interface bringup.launch.py arms:=true ee_type:=amazing_hand
ros2 launch openarm_test run_skills_only.launch.py
ros2 launch qvic_2026 qvic_2026.launch.py bt_xml_path:=/home/hans/universal_bot/Open_arm_a1_ws/install/qvic_2026/share/qvic_2026/config/qvic_2026_<name>_bt.xml
```
Check `config/hardware_config.yaml`'s (in `robot_hardware_interface`)
`control_mode` first: must be `position` (or `mit`) for BT-driven trajectory
replay to actually move the arm — `torque` mode (used for hand-guiding /
gravity comp) ignores position commands entirely.

## `sequence.yaml` conventions

```yaml
# Planning group indicator (key prefix): la/ra = arm, lh/rh = hand, head = head goal
# Key contains "Angle"  -> 7 joint values, replayed via PlanToJointTarget/PlanToJointSequence
# Key contains "Joint" (or anything else) -> pose data: 3 position + 4 quaternion
speed:
  <section>: <velocity>[, <acceleration>]   # MoveIt scaling factors [0-1], optional acceleration
```

- `*Angle` keys are joint-space (7 raw joint values) — this is what
  actually gets replayed.
- `*Joint` keys (or any non-`Angle` key) are Cartesian reference poses
  (x, y, z, qx, qy, qz, qw) — used as `--start`/`--end`/`--center`
  references by `trajectory_waypoint_generator`, not replayed directly.
- `speed:` entries scale velocity/acceleration per-section; missing or 0 =
  the skill's default profile speed.

## Collision safety

Neither recording nor formula-generation collision-checks anything by
itself. Real safety enforcement happens on **replay**: `PlanToJointSequence`
plans each segment individually (collision-aware) and the corner-blend step
re-validates the blended path with a safe fallback
(`control/motion_planner/src/moveit_cpp_planner_manager.cpp`,
`getJointSequence()` branch). One known gap either way:
`openarm_body_shell_link` has no collision geometry defined, so collision
checking against the torso shell itself is incomplete — another reason to
always watch the fake-hardware dry run before real hardware.

## Adding project-specific code

`src/` is empty so far. The rule of thumb used throughout this project:

- **Generic, reusable across projects** (new shape math, new FK/IK
  utilities, new yaml conventions) → a shared package under `control/`
  (see `control/trajectory_shapes`, `control/trajectory_waypoint_generator`,
  `control/waypoint_recorder`).
- **qvic_2026-specific** (custom choreography logic, competition-specific
  behavior) → a new `.cpp` in `src/`, wired into this package's own
  `CMakeLists.txt`/`package.xml` (`find_package(...)` +
  `add_executable(...)` + `ament_target_dependencies(...)`, same pattern
  every package in this workspace already uses).

To **combine several already-generated sequences** into one continuous run,
you likely don't need new C++ at all: each section gets its own
`<BehaviorTree ID="SectionName">` block from `sequence_to_bt`, and BT.CPP's
`<SubTree ID="OtherSectionName"/>` node lets a top-level `<Sequence>` chain
several of them together in one hand-composed XML. Only reach for a new
`.cpp` here if you need runtime logic (pick sequence A vs B, loop, react to
sensor state) beyond simple stitching.
