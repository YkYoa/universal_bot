# gesture_teleop

Human-arm gesture teleoperation for the OpenArm bimanual robot: MediaPipe
Pose is retargeted onto the two 7-DoF arms via a closed-form solver, behind
an explicit-arm dead-man-switch safety pipeline — plus a separate,
non-actuating browser mockup for watching the algorithm work with your own
webcam.

**Full design rationale, the kinematics findings, and the staged rollout
live in the project plan** (ask for it, or see git history for
`carefully-ssh-ubuntu-192-168-1-226-quirky-boot.md`). This file is the
practical "how do I run/test this" reference.

## Layout

```
gesture_teleop/
  core/              Pure Python + numpy. No rclpy/flask/mediapipe/cv2 -
                      see test_no_ros_in_core.py. This is the algorithm:
                      kinematics, filtering, retargeting, validation, the
                      GestureSession state machine.
  mp_runner.py        The ONLY module that imports mediapipe.
  mock_server.py      Non-actuating web mockup, Flask-SocketIO, port 5055.
  udp_link.py         Landmark datagram shared by mock_server.py (sender,
                      --enable-real only) and teleop_node.py (receiver).
  teleop_node.py      Real-arm rclpy node: UDP in, rate-limited joint
                      commands out, FSM-gated, dead-man-switched.
  command_sink.py     Pluggable output targets (Null/Recording/
                      ForwardPosition/Jtc/Preview).
  controller_switch.py  ros2_control controller-exclusivity switch.
web/                  The mockup's HTML/JS/CSS + vendored socket.io.
config/gesture_teleop.yaml   Reference tunables (see its own header).
launch/               ROS 2 launch files.
test/                 pytest suite - unit tests run with zero hardware/ROS;
                      test/integration/ needs Flask/Flask-SocketIO but
                      still no camera, no ROS, no mediapipe (MP_BACKEND=fake).
tools/gen_golden_fk.py  Regenerates test/fixtures/fk_golden.json from the
                      real robot's URDF via PyKDL - run manually on the
                      robot when the arm geometry changes.
```

## Running the tests (no hardware, no ROS required)

```bash
cd hardware/openarm/src/perception/gesture_teleop
python -m venv venv && source venv/bin/activate   # or your platform's equivalent
pip install numpy pytest hypothesis flask flask-socketio python-socketio

python -m pytest test -m "not integration" -q   # pure core + architectural invariants
python -m pytest test -m "integration" -q       # mock_server via Flask-SocketIO's test client
python -m pytest test -q                        # everything
```

`test_fk_against_golden` needs `test/fixtures/fk_golden.json`, which is
checked in. If the robot's arm geometry ever changes, regenerate it by
running `tools/gen_golden_fk.py` on the robot (it needs PyKDL, a system
package there) and copying the printed JSON back into that fixture.

## Running the mockup locally (no robot, no ROS)

```bash
cd hardware/openarm/src/perception/gesture_teleop
MP_BACKEND=real python -m gesture_teleop.mock_server --port 5055
# open http://localhost:5055/ and allow camera access
```

This needs a real `mediapipe` install (see the plan's dependency-isolation
section for why it lives in its own venv on the actual robot, not the
system Python) and `flask`/`flask-socketio`. It never touches ROS.

**`http://localhost:5055/` works over plain HTTP because `localhost` is
itself a secure context** - browsers only expose `navigator.mediaDevices`
(and therefore the camera) on `https://` or `http://localhost`/`127.0.0.1`.
Reached from another machine by LAN IP (`http://<robot-ip>:5055/`, the
robot's real deployment), that API is silently `undefined` and the page's
"Enable Camera" button will explain that HTTPS is required rather than
prompting for permission. Add `--https` to test that path locally too:

```bash
MP_BACKEND=real python -m gesture_teleop.mock_server --port 5055 --https
# open https://<your-lan-ip>:5055/ - accept the one-time self-signed-cert warning
```

`--https` generates a self-signed cert/key pair once under
`--tls-cert-dir` (`$GESTURE_TLS_CERT_DIR`, default
`/opt/openarm-gesture/tls`) via the system `openssl` binary and reuses it
on every subsequent start - see `tls_cert.py`. If `openssl` isn't on
`PATH`, it logs a warning and falls back to plain HTTP rather than
crashing the mockup over a TLS nicety.

## Offline replay (no camera, no browser, no ROS, no mediapipe)

```bash
python -m gesture_teleop.core.tools.replay test/fixtures/session.json \
    --side left --mirror-mode mirror --auto-engage
```

Prints the retargeted joint angles and FSM state for each frame of a
recorded/synthetic landmark session. Useful for regression-checking
retargeting changes against a fixed input.

## Deploying on the robot

See `deployment/systemd/openarm-gesture-mock.service` (mockup, safe to
enable any time) and `openarm-gesture-teleop.service` (real arm actuation
— **disabled by default**, only enable after the Stage 5 preview gate and
with an e-stop in hand, per the plan's staged rollout). Both read
`/etc/robot-healthmate/openarm.env` (see `deployment/config/openarm.env.example`
for the `GESTURE_*` variables).

The mock service's `ExecStart` passes `--https` (required for the camera
panel to work at all when reached by LAN IP - see the "Running the mockup
locally" section above) and needs `openssl` on `PATH`, which is already
part of Ubuntu's base install. It self-generates a cert on first start
under `/opt/openarm-gesture/tls` and reuses it after - `sudo rm -rf
/opt/openarm-gesture/tls && sudo systemctl restart
openarm-gesture-mock.service` forces a fresh one if it's ever needed.

The mockup runs in its own dependency-isolated venv at
`/opt/openarm-gesture/venv` (mediapipe pulls in `opencv-contrib-python`,
which must never be installed system-wide — it would shadow the Debian
`python3-opencv` that `cv_bridge`/`camera_bridge_node` depend on):

```bash
python3 -m venv --system-site-packages /opt/openarm-gesture/venv
/opt/openarm-gesture/venv/bin/pip install --no-deps mediapipe==0.10.14
/opt/openarm-gesture/venv/bin/pip install absl-py attrs flatbuffers numpy 'protobuf<5,>=4.25.3'
/opt/openarm-gesture/venv/bin/python -c "import mediapipe"   # verify before anything else

# mock_server.py's own deps (separate from mediapipe above - needed even to
# start the server). simple-websocket is required for a real WebSocket
# transport; without it Flask-SocketIO silently falls back to long-polling
# ("WebSocket transport not available" in the unit's journal) - the mockup
# would still load but frame streaming pays the polling round-trip/base64
# overhead the plan's transport section says to avoid.
/opt/openarm-gesture/venv/bin/pip install flask flask-socketio python-socketio simple-websocket
```

The pose model is already on disk on the robot at
`/opt/robot-healthmate-agent-os/current/ai-services/vision-api/models/pose_landmarker_lite.task`
— no need to re-download it (`GESTURE_POSE_MODEL_PATH` env var overrides
the path if it ever moves).

## Safety model, in one paragraph

Nothing moves without an explicit `engage()`, which itself only succeeds
when the corresponding arm is already `TRACKING` a good-quality person AND
the robot's own FSM (`sequence_executor_node`) reports `IDLE` AND the
`*_forward_position_controller` switch succeeds. Three independent timers
in different processes (client heartbeat, teleop_node's own landmark
watchdog, and the rate limiter's own hold ramp) are what make the
dead-man switch real rather than a convention — see `pipeline.py`'s and
`teleop_node.py`'s module docstrings for the full interlock list.
