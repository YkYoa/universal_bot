# OpenArm REST API — Android integration guide

Base URL `http://<robot-ip>:5050`, `Content-Type: application/json`.
Everything is CORS-open, no auth.

## 0.a Connecting

The backend now runs as a systemd service (`openarm-robot.service`) — it
starts automatically on power-on and restarts itself on failure. **You don't
need anyone to SSH in and launch anything.** After power-on, give it
30–60s to bring up MoveIt before expecting `200` from the API.

Which IP to use depends on the network:

| Scenario | Robot IP | Base URL |
|---|---|---|
| Shared WiFi (office/dev network) | `192.168.1.226` | `http://192.168.1.226:5050` |
| Pure local network, no WiFi/internet at all — tablet wired/switched to `end0` alongside the Jetson | `192.168.10.1` | `http://192.168.10.1:5050` |

If you don't know which network the tablet is on, try `GET /api/fsm/state`
against both; whichever returns `200` (not "connection refused"/timeout) is
the right one. A `503` means the service is up but the robot backend inside
it isn't ready yet — retry, don't treat it as a network error.

If the API is unreachable for more than ~2 minutes after power-on, that's a
real fault (not a "still booting" situation) — flag it, don't retry forever.

## 0. Safety

* **Power loss**: if the robot loses electrical power while a sequence is
  running, hit the physical emergency-stop button first. Once power is
  confirmed off, the arms have no holding torque and can be moved freely —
  pull each arm by hand back to its home position before power is restored,
  so the robot doesn't boot up with an unexpected pose as its "current"
  state.
* **Startup**: after every bringup (power-on or app reconnect), run
  `builtin:action_01` ("Home both arms") first, before anything else in the
  sequence list. Do not assume the arm is already at a known pose just
  because the app is showing `IDLE`.
* **Before switching actions**: always run `builtin:action_01` again before
  starting a *different* action or sequence, not just at startup. Every
  other action/sequence assumes it starts from home - going straight from
  one action into another without homing in between can plan a valid but
  unexpected path between two arbitrary poses.

There are three groups of endpoints:

| Group | What it does | Served by |
|---|---|---|
| **FSM** | Watch and control the state machine | always |
| **Store** | Build and edit sequences (the teaching pendant) | only when `qvic_2026` is installed |
| **Direct** | Move one joint/group right now, read poses | always |

---

## 1. The state machine

Two layers, published together. `robot_state` is what to show big; the
sequence fields are the detail underneath it.

```
robot_state     BOOTING | IDLE | RUNNING | PAUSED | FAULT | ESTOP | TEACHING
sequence_state  (empty) | LOADING | VALIDATING | STEP_PLANNING | STEP_EXECUTING
                | STEP_DONE | LOOP_CHECK | COMPLETED | FAILED | CANCELLED
```

### Live state — use the socket, not polling

Connect with Socket.IO and listen for **`fsm_state`**. One event per
transition; nothing is sent while the robot is idle.

```json
{
  "stamp": 1786351177.66,
  "robot_state": "RUNNING",
  "sequence_name": "qvic_2026_both",
  "sequence_state": "STEP_EXECUTING",
  "step_index": 2, "step_total": 3,
  "step_name": "Play waveEllipse", "step_type": "move_joint_sequence",
  "loop_index": 0, "loop_total": -1,
  "control_mode_active": "position",
  "progress": 0.33,
  "fault_reason": ""
}
```

`loop_total: -1` means it loops forever. `progress` is 0.0–1.0 across the
whole run.

`GET /api/fsm/state` returns the same object for a cold start
(`{"success": true, "state": {...}}`). A 503 means the executor is not
running.

### The diagram

`GET /api/fsm/graph` returns both layers as nodes and edges, each node with
`id`, `label`, `kind` (`start|normal|active|success|warning|error|special`)
and a description. Build the diagram from this rather than hardcoding the
state list — it comes from the executor's own config file, so it cannot drift.
The web viewer at `/dashboard/fsm.html` is a worked example.

### Controls

`POST /api/fsm/command` `{"command": "..."}`

| command | effect |
|---|---|
| `pause` | stops at the **next step boundary**; a trajectory already moving finishes |
| `resume` | continues from where it paused |
| `step` | runs exactly one step, then holds (only while paused) |
| `cancel` | stops now — cancels the in-flight motion goal |
| `estop` | cancels everything and parks in `ESTOP` |
| `clear_fault` | `FAULT`/`ESTOP` → `IDLE` |
| `enter_teach` / `exit_teach` | hand-guiding; refused unless the arm is in torque mode |

`200` = done, `409` = refused with a human-readable `message`
("not paused", "no fault to clear"). Show the message.

`POST /api/stop` is a shortcut for `estop`.

### Running something

`POST /api/sequence/run`

```json
{"name": "qvic_2026_both", "repeat": 1, "velocity": 0.0, "dry_run": false}
```

* `repeat` — `0` uses the sequence's own setting, `-1` loops forever
* `velocity` — `0` uses the sequence's own setting
* `dry_run` — validates and walks every step **without moving**. Use this to
  check a sequence the user just built.

Returns as soon as the goal is accepted. **Progress comes over the socket**,
not in this response. A `409` means the robot is busy, faulted, or teaching.

`GET /api/actions` lists the hardcoded actions. Run one the same way,
with `{"name": "builtin:action_01"}`.

| id | label | what it does |
|---|---|---|
| `action_01` | Home both arms | always run this first, and before switching to a different action |
| `action_02` | (left ellipse wave) | loops until cancelled |
| `action_03` | Head rotate | loops left/right until cancelled |
| `action_04` | Wave left arm | loops until cancelled |
| `action_05` | Wave right arm | loops until cancelled |
| `action_06` | Loop right arm | loops until cancelled |
| `action_07` | Loop left arm | loops until cancelled (mirror of action_06) |
| `action_08` | (right ellipse wave) | loops until cancelled |
| `action_09` | Show | both arms to the show pose, one move, then done |
| `action_10` | Head rotate left | head to +10°, one move, then done |
| `action_11` | Head rotate right | head to -10°, one move, then done |
| `action_12` | Head rotate home | head to 0°, one move, then done |

`action_10`/`11`/`12` are the ones to use for discrete left/right/home
buttons in the app - `action_03` is a different thing (an endless sweep,
only stoppable with `cancel`).

---

## 2. Control mode — read this before building the editor

`GET /api/control-mode`

```json
{"control_mode": "position", "runnable_step_modes": ["any", "position|mit"],
 "switchable": false}
```

The arm's control mode is **fixed when the hardware boots** — the motor
register is written once during init. There is no runtime switch, and the API
does not offer one.

What this means for the UI:

* Every step type declares a `control_mode` (`any`, `position|mit`, or
  `torque`). Grey out the ones that don't match `runnable_step_modes`.
* A sequence cannot mix `position|mit` and `torque` steps — the store rejects
  that at edit time with an explanation.
* Starting a sequence whose mode doesn't match fails in `VALIDATING`, before
  the arm moves, and `fault_reason` says what to change.

`torque` mode is what hand-guiding needs; `position`/`mit` is what replaying a
trajectory needs.

---

## 3. Building sequences (teaching pendant)

Only present when `qvic_2026` is installed. `GET /api/store/info` tells you.

### The palette

`GET /api/step-types` returns every step type with its fields, types,
defaults, and required control mode. **Build the drag-and-drop palette from
this** — do not hardcode a list.

```json
{"step_types": {
  "move_joint": {
    "label": "Move to waypoint",
    "description": "...",
    "control_mode": "position|mit",
    "fields": [
      {"name": "arm", "type": "enum", "required": true,
       "options": ["left_arm", "right_arm", "both_arms"]},
      {"name": "waypoint", "type": "waypoint_ref", "required": false, "...": "..."}
    ]
  }
}}
```

Field `type` values: `enum` (has `options`), `string`, `bool`, `float`
(may have `minimum`/`maximum`), `float[]`/`int[]`/`string[]` (may have
`length`), `waypoint_ref` (a `section/name` string), `section_ref`.

The types available today:

| type | what it does |
|---|---|
| `move_joint` | one arm to a stored waypoint (or raw joint values) |
| `move_joint_sequence` | replay a whole waypoint section as one blended motion |
| `move_pose` / `named_pose` | Cartesian target / SRDF named pose |
| `hand_pose` | amazing_hand yaw/flex, both hands together |
| `hand_fingers` | the 4-finger hand board, over its own REST API (see below) |
| `move_groups` | **arms + head + hands together, in any combination** (see below) |
| `gripper` | open/close the parallel gripper |
| `wait` | hold for N seconds |
| `add_object` / `remove_object` | put an obstacle in the planning scene |
| `attach_object` / `detach_object` | grasp / release — see below |
| `allow_collision` / `disallow_collision` | direct collision-matrix entries |
| `set_speed` | change scaling for the following steps |
| `teach_hold` | stay compliant so a human can move the arm (torque mode only) |

### CRUD

| Method | Path | Notes |
|---|---|---|
| GET | `/api/sequences` | list with step counts and required mode |
| POST | `/api/sequences` | `{name, arm, planner_profile, repeat, steps?}` |
| GET/PUT/DELETE | `/api/sequences/<name>` | PUT accepts `steps` and `new_name` |
| POST | `/api/sequences/<name>/duplicate` | `{new_name}` |
| POST | `/api/sequences/<name>/steps` | `{type, params, index?}` — `index` inserts |
| PUT/DELETE | `/api/sequences/<name>/steps/<i>` | edit or remove one step |
| POST | `/api/sequences/<name>/reorder` | `{"order": [2,0,1]}` — current indices in their new order |

`400` with a precise `message` on a bad step
(`"move_joint: set exactly one of 'waypoint' or 'positions'"`). Surface it —
the validator is the same one the robot uses.

### Two hands, two different things

`hand_pose` drives the **amazing_hand** through ros2_control.
`hand_fingers` drives the **CH32V307 4-finger board** through its own REST API
on port 5051. They are different hardware on different transports — one does
nothing to the other. `GET /api/capabilities` on the gateway says which is
actually present.

`hand_fingers` takes 8 angles in degrees, `F1m1, F1m2, F2m1 … F4m2`:

```json
{"type": "hand_fingers", "params": {"fingers": [90,45, 90,45, 90,45, 90,45]}}
```

`{"home_first": true}` homes every finger first; on its own it just homes.
The board clamps each motor to its own safe range, so the applied value can
differ from the requested one.

A sequence containing `hand_fingers` is refused at `VALIDATING` when the board
is not reachable, with the reason — nothing moves.

### Moving several groups at once

`move_groups` is one step that fires **every subsystem you name, at the same
time**. Which fields you fill in decides which groups take part, so the single
type covers arms+hands, arms+head, hands+head, all three, and any other
combination — there is no separate step type per pairing.

```json
{"type": "move_groups", "params": {
  "arm": "both_arms",
  "waypoint": "homePoses/laHomeAngle",
  "right_waypoint": "homePoses/raHomeAngle",
  "head": [0.0, -0.2],
  "fingers": [90,45, 90,45, 90,45, 90,45],
  "duration": 1.5
}}
```

| field group | drives |
|---|---|
| `arm` + `waypoint`/`right_waypoint`/`positions` | one arm goal |
| `arm` + `section`/`right_section` | replay a whole section instead |
| `head` | `[pan, tilt]` in radians, through `head_controller` |
| `fingers` | the 4-finger board (8 values) |
| `left_yaw`/`left_flex`/`right_yaw`/`right_flex` | amazing_hand |

The step finishes when the slowest group finishes; the first failure is what
gets reported. Both arms always move through a single `both_arms` goal — MoveIt
plans one trajectory per group, so two separate arm goals would be serialised
by the skill server rather than overlapping.

### Waypoints

`GET /api/waypoints` returns every waypoint plus a `sections` summary.
Names repeat across sections, so a step always refers to one as
**`section/name`** (`homePoses/laHomeAngle`).

`POST /api/waypoints` records one. `{"source": "live"}` captures where the arm
is right now — the equivalent of the `record_waypoint` terminal tool, which is
how a teach-and-record flow works from the tablet:

```json
{"name": "laPick1Angle", "section": "pickPoses", "arm": "left_arm", "source": "live"}
```

`DELETE /api/waypoints/<section>/<name>`.

---

## 4. Collision and grasping

Every motion is collision-checked by MoveIt already; a plan that would hit
something fails rather than moving.

To pick something up, the sequence is: put the object in the scene, move to
it, then **attach** it.

```
POST /api/scene/add     {"object_id":"box1", "primitive":"box",
                         "dimensions":[0.05,0.05,0.05], "position":[0.4,0.2,1.1]}
POST /api/scene/attach  {"object_id":"box1", "link":"openarm_left_hand_tcp"}
POST /api/scene/detach  {"object_id":"box1"}
```

Attaching is what "allow collision while grasping" means in practice: MoveIt
carries the object with the arm, counts it as part of the robot when avoiding
everything else, and stops reporting the fingers touching it as a collision.
Leave `touch_links` out and the whole hand is used.

Use `POST /api/scene/allow` `{"object_id":"box1","links":["table"]}` only for
the surface an object rests on. `POST /api/scene/clear` resets the scene.

The same operations exist as step types, so a stored sequence can grasp
without the app driving it step by step.

---

## 5. Direct control (settings / calibration screens)

Groups: `left_arm` (7), `right_arm` (7), `both_arms` (14),
`left_hand_fingers` (8), `right_hand_fingers` (8). Joint names and order:
`GET /api/docs` → `planning_groups`.

* `POST /api/move/joint` — one joint, e.g.
  `{"group":"left_arm","joint":"openarm_left_joint4","value":45,"unit":"deg"}`
* `POST /api/move/joints` — a whole group; `POST /api/plan/joints` is the same
  call without moving (reachability / collision check)
* `POST /api/move/named` — SRDF named pose
* `POST /api/move/pose`, `POST /api/move/workspace` — Cartesian
* `POST /api/gripper`, `POST /api/hand` — end effectors
* `GET /api/status`, `GET /api/pose/<group>` — read back

**`head` is NOT a usable group in `/api/move/joints`** (confirmed 2026-08-18
— fails with `PLANNING_FAILED`): the head is driven directly by
`head_controller`, not through a MoveIt planning group at all. Use its own
endpoint instead:

`POST /api/head` — `{"action": "left"}` / `{"action": "right"}` /
`{"action": "home"}`, or raw `{"pan": 0.0, "tilt": 0.1745, "duration": 0.4}`.
This is what the discrete left/right/home buttons should call — same effect
as `builtin:action_10`/`11`/`12`, just without going through the FSM/sequence
layer, so use whichever fits how the button is wired.

Values are radians unless you add `"unit": "deg"`.
Live joint angles stream over Socket.IO: emit `subscribe_joint_states`, listen
for `joint_states` (10 Hz).

---

## 6. Error handling, in one place

| code | meaning | what to show |
|---|---|---|
| 400 | bad request body | the `message` — it names the field |
| 409 | refused because of robot state | the `message` — "already running", "not paused" |
| 422 | planning or execution failed | the `message` |
| 503 | executor or robot_skills not running | "robot offline" |

A sequence that fails at runtime does not come back as an HTTP error — it
arrives as an `fsm_state` event with `robot_state: "FAULT"` and a
`fault_reason`. Show that, and offer `clear_fault`.
