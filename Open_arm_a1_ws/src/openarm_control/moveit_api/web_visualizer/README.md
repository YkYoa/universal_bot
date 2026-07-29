# OpenArm 3D Web Dashboard

Browser-based 3D view of the robot, driven by live joint states — no
RViz, no monitor/X server needed on the robot's own machine (the IQ-9075
board). Open it from any phone or PC on the same network as the robot.

## Open it

```
http://<robot-ip>:5050/dashboard/
```

It's served by `robot_api_server` (same Flask process as the REST API,
same port 5050) — nothing extra to run beyond the usual:
```bash
ros2 launch moveit_api robot_api.launch.py
```

## How it works

* `GET /api/urdf` returns the live `robot_description` (robot_state_publisher
  latches it, `moveit_ee_controller.py` subscribes once and caches it).
* `GET /packages/<pkg>/<path>` resolves the URDF's `package://` mesh URIs
  by serving files straight out of that ROS package's installed share
  directory (read-only, only for packages actually installed).
* The page loads the URDF with [urdf-loader](https://github.com/gkjohnson/urdf-loaders)
  on top of [three.js](https://threejs.org/), then drives joint angles
  live via the same `joint_states` Socket.IO stream `robot_api_server.py`
  already emits at 10Hz (`subscribe_joint_states` event — see README.md
  one level up).
* The sequence dropdown/Run/Stop buttons call the existing
  `GET /api/sequences`, `POST /api/sequence`, `POST /api/stop` endpoints —
  no separate control path.
* "▸ joint control" is a per-group RViz2 MotionPlanning-panel workflow:
  one slider per joint (from `GET /api/docs`' `planning_groups` crossed
  with each joint's real `<limit>` from the loaded URDF), plus **Plan** /
  **Execute** / **↺ reset** for that group.
  - Dragging a slider only moves the on-screen model — nothing is sent
    over the network, and the group stops following the live
    `joint_states` stream (it's now showing a candidate target, not the
    live robot).
  - **Plan** dry-runs that target via the new `POST /api/plan/joints`
    (MoveGroup `plan_only`, no motion) — warns inline if MoveIt can't find
    a valid plan (collision/unreachable/etc), otherwise unlocks Execute.
  - **Execute** commits the same target via the existing
    `POST /api/move/joints`; on success the group resumes following the
    live stream.
  - **↺** discards the candidate target and snaps sliders back to the
    robot's last known real position, without moving anything.

## Rendering

Shadow-mapped lighting (key/fill/rim + a shadow-catching floor), ACES
filmic tone mapping, and a procedural `RoomEnvironment` (PMREM, no
external HDRI file) for PBR ambient/reflections. `urdf-loader` always
builds `MeshPhongMaterial` regardless of URDF content — `upgradeMaterials()`
in `app.js` swaps those for `MeshStandardMaterial` after each mesh loads
(Phong ignores `scene.environment` entirely, so without this swap the
environment lighting would have no visible effect on the robot).

## Files

```
index.html   - page shell, import map, panel UI
app.js       - three.js scene, URDF loading, joint streaming, sequence controls
vendor/      - three.js, urdf-loader, socket.io-client - vendored (not CDN)
             so the dashboard still works with no internet on the LAN.
```

## Updating vendored libraries

Everything under `vendor/` is fetched straight from unpkg/the socket.io
CDN, unmodified. To bump a version, re-download the same files:

| File | Source |
|---|---|
| `vendor/three/three.module.js` | `unpkg.com/three@<ver>/build/three.module.js` |
| `vendor/three/examples/jsm/loaders/{STL,Collada,TGA}Loader.js` | `unpkg.com/three@<ver>/examples/jsm/loaders/...` |
| `vendor/three/examples/jsm/controls/OrbitControls.js` | `unpkg.com/three@<ver>/examples/jsm/controls/OrbitControls.js` |
| `vendor/three/examples/jsm/environments/RoomEnvironment.js` | `unpkg.com/three@<ver>/examples/jsm/environments/RoomEnvironment.js` |
| `vendor/urdf-loader/{URDFLoader,URDFClasses}.js` | `unpkg.com/urdf-loader@<ver>/src/...` |
| `vendor/socket.io/socket.io.min.js` | `cdn.socket.io/<ver>/socket.io.min.js` — must match the `python-socketio` server's Engine.IO protocol version (currently v4.x client for `python-socketio` 5.x server) |

`urdf-loader`'s peer dependency is `three>=0.152.0` — keep both in step.
