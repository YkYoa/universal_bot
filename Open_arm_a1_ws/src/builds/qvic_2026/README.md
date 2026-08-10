# qvic_2026

Project home for the QVIC 2026 (Qualcomm) competition entry. Holds this
project's sequence store, its ten hardcoded actions, and its launch setup.
Reuses the generic state machine from `control/sequence_executor` rather than
duplicating it.

## What runs

```
Android app / web page
        │  HTTP + Socket.IO :5050        (moveit_api/robot_api_server)
        ▼
   FSM bridge  ──ROS──▶  qvic_fsm_node   ──action──▶  robot_skills  ──▶ MoveIt
                          │                            │
                          ├─ SQLite store              └─ scene_command
                          │  (data/sequences.db)          (collision, grasp)
                          └─ 10 builtin actions
```

`qvic_fsm_node` is `sequence_executor`'s state machine with two things added:
a SQLite `SequenceSource` reading the store the Android app edits, and this
project's `BuiltinActionRegistry` entries. Everything else - the supervisory
FSM, the step walker, validation, pause/resume/cancel - is the shared package.

## The store

`data/sequences.db` is the runtime source of truth. `config/sequence.yaml`
still exists because `waypoint_recorder` and `trajectory_waypoint_generator`
write it, and because it is what git diffs - but the executor reads the store.

Seed or re-sync it:

```
ros2 run qvic_2026 sequence_store_cli.py import --file src/builds/qvic_2026/config/sequence.yaml
ros2 run qvic_2026 sequence_store_cli.py list
ros2 run qvic_2026 sequence_store_cli.py export --file /tmp/roundtrip.yaml
```

Import is lossless in both directions for everything the flat YAML schema can
express. Steps it cannot express (waits, scene edits, per-step speed changes)
are dropped on export, and the CLI says so.

A **sequence** is an ordered list of **steps**. That is the change from the old
schema, which had one fixed shape - a home step, then a body looped N times.
The YAML importer unrolls that shape into steps, so old entries keep working.

Waypoint names repeat across sections (`laHomeAngle` is in both `homePoses`
and `waveHome`), so a step always refers to one as **`section/name`**.

## Control mode

The arm's control mode is fixed when the hardware boots - `hardware_config.yaml`
is read once and written to the motor register during init. There is no runtime
switch.

Every step declares the mode it needs (`any`, `position|mit`, or `torque`), and
the FSM checks all of them in `VALIDATING` before anything moves. A mismatch
fails with a message naming the step, instead of `torque` mode silently
ignoring position commands the way it used to.

The store refuses to put `torque` and `position|mit` steps in one sequence at
edit time, since no single boot can satisfy both.

## Running it

```
ros2 launch qvic_2026 qvic_2026.launch.py arm:=left use_rviz:=true
```

| argument | meaning |
|---|---|
| `arm:=left\|right\|both` | picks `qvic_2026_left/_right/_both` |
| `sequence:=<name>` | any other sequence by name |
| `autostart:=false` | sit in IDLE and wait for the app or the web page |
| `use_api:=true` | also launch the REST/WebSocket API on :5050 |
| `use_db:=false` | read `config/sequence.yaml` instead of the store |
| `use_rviz:=true` | RViz |

With `use_api:=true`: the state-machine viewer is at
`http://<robot-ip>:5050/dashboard/fsm.html`, the 3D dashboard at
`/dashboard/`. The Android contract is
`control/moveit_api/android_api_guide.md`.

Driving it by hand:

```
ros2 topic echo /sequence_executor_node/state
ros2 action send_goal /sequence_executor_node/run_sequence openarm_messages/action/RunSequence "{sequence_name: qvic_2026_left, repeat_override: 1}"
ros2 service call /sequence_executor_node/fsm_command openarm_messages/srv/FsmCommand "{command: pause}"
```

`dry_run: true` on the goal walks and validates every step without sending a
single motion goal - the cheapest way to check a sequence the app just built.

## The ten hardcoded actions

`src/qvic_actions.cpp` holds `action_01` .. `action_10`. `action_01` is
implemented as a worked example; the rest fail with "has no body yet" until
filled in. They appear alongside stored sequences as `builtin:<id>` and run
through the same path.

The file's header comment covers the three rules that matter: call `done`
exactly once on every path, check `ctx.cancelled()` between async hops, and
declare the control mode honestly.

## Recording waypoints

Unchanged - both tools still write `config/sequence.yaml`, then you import:

```
ros2 run waypoint_recorder record_waypoint --loop --file src/builds/qvic_2026/config/sequence.yaml
ros2 run trajectory_waypoint_generator generate_waypoint_sequence --arm la --section waveEllipse --shape ellipse --start key:laPreWaveJoint --end key:laEndWaveJoint --file src/builds/qvic_2026/config/sequence.yaml --num-points 20 --height-ratio 0.3
```

`--section` must not end in a digit - the section digit and the point number
would concatenate ambiguously.

The API can also record from the tablet: `POST /api/waypoints` with
`{"source": "live"}` captures the arm's current position.

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
