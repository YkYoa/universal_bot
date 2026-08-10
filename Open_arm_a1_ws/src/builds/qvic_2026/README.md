# qvic_2026

Project home for the QVIC 2026 (Qualcomm) competition entry. Holds this
project's own `sequence.yaml` and launch setup. Reuses generic tooling
from `control/` rather than duplicating it — this package should stay
thin.

## Directory layout

```
qvic_2026/
├── config/
│   └── sequence.yaml           # waypoints/poses AND the sequences: entries (source of truth)
├── launch/
│   └── qvic_2026.launch.py     # picks a sequences: entry (arm:= or sequence:=), launches sequence_executor
├── src/                        # project-specific C++ (empty so far - see "Adding project code" below)
├── include/
├── CMakeLists.txt
└── package.xml
```

## The pipeline, end to end

Every sequence this project runs goes through three stages — no
generation/build step in between: `sequence_executor_node` reads
`config/sequence.yaml` directly off disk at launch time.

**1. Get waypoints into `config/sequence.yaml`** — two ways:

- **Hand-guide + record** (`control/waypoint_recorder`): put the arm in
  `control_mode: torque` (gravity compensation active), physically move it,
  and record positions live.
  ```
  ros2 run waypoint_recorder record_waypoint --loop --file /home/hans/universal_bot/Open_arm_a1_ws/src/builds/qvic_2026/config/sequence.yaml
  ```
  Interactive: pick arm (`la`/`ra`/`both`), pick/create a section, then
  press Enter per waypoint (auto-numbered `<arm><Section><N>Angle`), `s` to
  switch section, `q` to finish.

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

**2. Define (or reuse) a `sequences:` entry.** Each entry is a flat dict
under the top-level `sequences:` key in `sequence.yaml`:

| field | meaning |
|---|---|
| `arm` | `left_arm` \| `right_arm` \| `both_arms` |
| `planner_profile` | planner profile name (e.g. `fast_ptp`, `realtime_rrt`) |
| `home_section` | *(optional)* section run once before the loop — arm target plus, if present, `lh`/`rhHomeYaw`/`Flex` hand hold |
| `body_section` | a section replayed as the sequence body (`PlanToJointSequence`) |
| `body_right_section` | *(optional)* paired right-arm section, interleaved 1:1 with `body_section` |
| `body_sections` | *(optional, comma-list, alternative to `body_section`)* section names cycled through in order — used for hand-pose-only sequences |
| `repeat` | `-1` = infinite, default `1` |
| `exclude_points` | *(optional, comma-list, 1-indexed)* waypoints to skip (workaround for a known-bad point) |

Example (arm wave with a home step, looping forever):
```yaml
sequences:
  qvic_2026_both:
    arm: both_arms
    planner_profile: realtime_rrt
    home_section: homePoses
    body_section: waveEllipse
    body_right_section: waveEllipseR
    repeat: -1
```
Example (hand-only animation, arm holds still):
```yaml
sequences:
  hand_open_close:
    arm: both_arms
    home_section: homePoses
    body_sections: handOpen, homePoses
    repeat: -1
```
`velocity`/`acceleration` scaling is still looked up from the existing
top-level `speed:` dict by section name — not a `sequences:` field.

**3. Replay — fake hardware first, always:**
```
ros2 launch qvic_2026 qvic_2026.launch.py arm:=left use_rviz:=true
```
`arm:=left|right|both` maps to `qvic_2026_left`/`_right`/`_both` in
`sequences:`. For any other entry (e.g. `hand_open_close`), pass it
directly and skip `arm:=`:
```
ros2 launch qvic_2026 qvic_2026.launch.py sequence:=hand_open_close use_rviz:=true
```
This is a fully self-contained fake-hardware stack (robot_state_publisher,
fake ros2_control, move_group, robot_skills_node, sequence_executor, RViz)
— one command, no other terminals needed. Add `use_api:=true` to also
launch `bt_viewer`'s web UI (http://127.0.0.1:5000) for live sequence
status. Confirm the motion looks right in RViz *before* touching real
hardware. Editing `sequence.yaml` and relaunching is enough to see a
change — no `colcon build` needed unless you touched C++.

**Real hardware** (only after the fake-hardware dry run looks right), 3
terminals:
```
ros2 launch robot_hardware_interface bringup.launch.py arms:=true ee_type:=amazing_hand
ros2 launch openarm_test run_skills_only.launch.py
ros2 launch qvic_2026 qvic_2026.launch.py arm:=both
```
Check `config/hardware_config.yaml`'s (in `robot_hardware_interface`)
`control_mode` first: must be `position` (or `mit`) for trajectory replay
to actually move the arm — `torque` mode (used for hand-guiding / gravity
comp) ignores position commands entirely.

## `sequence.yaml` conventions

```yaml
# Planning group indicator (key prefix): la/ra = arm, lh/rh = hand, head = head goal
# Key contains "Angle"  -> 7 joint values, replayed via joint_target/joint_sequence
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
- `sequences:` (see above) is additive to all of this — it never changes
  how existing data sections are written or read.

## Collision safety

Neither recording nor formula-generation collision-checks anything by
itself. Real safety enforcement happens on **replay**: joint-sequence
planning plans each segment individually (collision-aware) and the
corner-blend step re-validates the blended path with a safe fallback
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

To **combine several already-defined sequences** into one continuous run,
you likely don't need new C++ at all: `home_section`/`body_sections` in a
single `sequences:` entry already covers "run this once, then cycle
through these." Only reach for a new `.cpp` here if you need runtime logic
(pick sequence A vs B, react to sensor state) beyond what the flat schema
expresses.
