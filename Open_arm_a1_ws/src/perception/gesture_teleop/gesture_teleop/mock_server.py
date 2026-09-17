"""Non-actuating gesture-follow mockup server, port 5055 by default.

Structural guarantee (proven, not just documented, by
test_mock_never_commands.py's AST walk): this module imports no rclpy, no
command_sink, and never names the real position-command controller topic -
the ONLY bridge from this process toward the real robot is an OPTIONAL
localhost UDP socket, opened only when started with --enable-real, which
sends bare landmark data (not commands) for teleop_node to independently
retarget/validate/actuate on its own. Absent --enable-real, this process
cannot drive the arm even if every other safety layer in teleop_node were
deleted.

Per-session (per Socket.IO sid) state, not a global: each browser tab gets
its own PoseLandmarker instance and its own pair of GestureSession objects
(left/right robot arm mockups), created on connect and torn down on
disconnect. This is the deliberate fix for the `_streaming` global-flag
anti-pattern in moveit_api/robot_api_server.py (robot_api_server.py:1237),
where one client's disconnect silently kills every other client's stream -
see test_mock_server.py's regression tests for both that and the
"broadcasts landmarks to the wrong client" sibling bug (always emit with
room=sid, never a bare socketio.emit()).
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO

from gesture_teleop import mp_runner
from gesture_teleop import tls_cert
from gesture_teleop import udp_link
from gesture_teleop.core import frames as fr
from gesture_teleop.core import robot_model as rm
from gesture_teleop.core.landmarks import ARM_TORSO_CONNECTIONS
from gesture_teleop.core.pipeline import GestureSession

MAX_CONCURRENT_SESSIONS = 2
IDLE_REAP_SECONDS = 60.0
IDLE_REAP_CHECK_PERIOD_S = 15.0

app = Flask(__name__)
app.config["SECRET_KEY"] = "gesture-teleop-mock-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", manage_session=False)

_enable_real = False
_udp_target: Optional[tuple] = None
_udp_socket: Optional[socket.socket] = None


def _web_dir() -> str:
    """Prefer the ament-installed share dir when running as a ROS package
    (matches moveit_api/robot_api_server.py's _web_visualizer_dir()
    pattern); fall back to the source tree for local/laptop runs (plan
    Stage 3: "runs on a laptop against your own webcam, robot untouched,
    service not even installed")."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(get_package_share_directory("gesture_teleop"), "web")
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


@dataclass
class _Session:
    sid: str
    landmarker: mp_runner.LandmarkerBase
    mirror_mode: str = "mirror"
    left: GestureSession = field(init=False)
    right: GestureSession = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_activity: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        self.left = GestureSession(side="left", mirror_mode=self.mirror_mode)
        self.right = GestureSession(side="right", mirror_mode=self.mirror_mode)

    def set_mirror_mode(self, mode: str) -> None:
        self.mirror_mode = mode
        self.left.mirror_mode = mode
        self.right.mirror_mode = mode


_sessions: dict[str, _Session] = {}
_sessions_lock = threading.Lock()


def _arm_payload(target) -> dict:
    frames = rm.forward_kinematics(target.side, target.q)
    points = [rm.shoulder_reference(target.side).tolist()] + [p.tolist() for p in frames.origins]
    return {
        "side": target.side,
        "valid": target.valid,
        "reason": target.reason.value,
        "q": target.q.tolist(),
        "points": points,  # [shoulder, joint1..joint7 hinge positions] for the drawn-arm mockup
        "elbow_theta": target.elbow_theta,
        "gimbal_band": target.gimbal_band,
    }


def _skeleton_payload(skeleton) -> dict:
    if skeleton is None:
        return {"landmarks": None, "connections": ARM_TORSO_CONNECTIONS}
    landmarks = {
        str(idx): {"x": p.xyz[0], "y": p.xyz[1], "visibility": p.visibility}
        for idx, p in skeleton.image_landmarks.items()
    }
    return {"landmarks": landmarks, "connections": ARM_TORSO_CONNECTIONS}


def _maybe_send_udp(skeleton) -> None:
    """Only reached when --enable-real was passed at startup AND a socket
    was successfully opened - see main(). Sends raw landmark data (never a
    joint command, see udp_link.py's module docstring) for teleop_node to
    independently process."""
    if not _enable_real or _udp_socket is None or skeleton is None:
        return
    udp_link.send_skeleton(_udp_socket, _udp_target, skeleton)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(_web_dir(), "gesture.html")


@app.route("/<path:filename>", methods=["GET"])
def static_files(filename):
    return send_from_directory(_web_dir(), filename)


@app.route("/health", methods=["GET"])
def health():
    with _sessions_lock:
        n = len(_sessions)
    return {
        "status": "ok",
        "sessions": n,
        "max_sessions": MAX_CONCURRENT_SESSIONS,
        "enable_real": _enable_real,
    }, 200


@socketio.on("connect")
def ws_connect():
    sid = request.sid
    with _sessions_lock:
        if len(_sessions) >= MAX_CONCURRENT_SESSIONS:
            socketio.emit("error", {"message": "too many concurrent sessions"}, room=sid)
            return False
        try:
            landmarker = mp_runner.create_landmarker()
        except mp_runner.PoseLandmarkerUnavailable as exc:
            socketio.emit("error", {"message": f"pose model unavailable: {exc}"}, room=sid)
            return False
        _sessions[sid] = _Session(sid=sid, landmarker=landmarker)
    return True


@socketio.on("disconnect")
def ws_disconnect():
    sid = request.sid
    with _sessions_lock:
        session = _sessions.pop(sid, None)
    if session is not None:
        session.landmarker.close()


@socketio.on("set_mirror_mode")
def ws_set_mirror_mode(data):
    sid = request.sid
    mode = (data or {}).get("mode", "mirror")
    if mode not in ("mirror", "same_side"):
        return
    with _sessions_lock:
        session = _sessions.get(sid)
    if session is not None:
        with session.lock:
            session.set_mirror_mode(mode)


@socketio.on("engage")
def ws_engage(data):
    sid = request.sid
    with _sessions_lock:
        session = _sessions.get(sid)
    if session is None:
        return
    now = time.monotonic()
    with session.lock:
        ok_l, why_l = session.left.engage(now)
        ok_r, why_r = session.right.engage(now)
    socketio.emit("engage_result", {"left": {"ok": ok_l, "reason": why_l},
                                     "right": {"ok": ok_r, "reason": why_r}}, room=sid)


@socketio.on("disengage")
def ws_disengage(_data):
    sid = request.sid
    with _sessions_lock:
        session = _sessions.get(sid)
    if session is None:
        return
    now = time.monotonic()
    with session.lock:
        session.left.disengage(now)
        session.right.disengage(now)


@socketio.on("heartbeat")
def ws_heartbeat(_data):
    """Dead-man heartbeat while the operator holds the engage control in
    the browser - see plan interlock 3. A key-hold alone is not a
    dead-man: the CLIENT is responsible for also sending this on an
    interval while held and for immediately calling 'disengage' on
    blur/visibilitychange/pagehide (see web/gesture.js)."""
    sid = request.sid
    with _sessions_lock:
        session = _sessions.get(sid)
    if session is None:
        return
    now = time.monotonic()
    with session.lock:
        session.left.heartbeat(now)
        session.right.heartbeat(now)


@socketio.on("frame")
def ws_frame(data):
    """Binary JPEG frame from the browser (see web/gesture.js - sent via
    canvas.toBlob()/arrayBuffer(), never a base64 data URL: python-socketio
    handles bytes as a native binary attachment with zero base64 inflation,
    per the plan's transport guidance)."""
    sid = request.sid
    with _sessions_lock:
        session = _sessions.get(sid)
    if session is None:
        return

    import cv2

    if not session.lock.acquire(blocking=False):
        return  # a previous frame is still being processed - drop (latest-wins backpressure)
    try:
        session.last_activity = time.monotonic()
        buf = np.frombuffer(data, dtype=np.uint8)
        frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return
        now = time.monotonic()
        skeleton = session.landmarker.detect(frame_bgr, now)
        _maybe_send_udp(skeleton)

        torso_frame = None
        torso_reason = None
        if skeleton is not None:
            torso_frame, torso_reason = fr.build_torso_frame(skeleton)

        left_target = session.left.process_frame(skeleton, torso_frame, now)
        right_target = session.right.process_frame(skeleton, torso_frame, now)

        payload = {
            "skeleton": _skeleton_payload(skeleton),
            "torso_ok": torso_frame is not None,
            "torso_reason": torso_reason.value if torso_reason is not None else None,
            "arms": {"left": _arm_payload(left_target), "right": _arm_payload(right_target)},
            "state": {"left": session.left.state.value, "right": session.right.state.value},
            "mirror_mode": session.mirror_mode,
        }
        socketio.emit("result", payload, room=sid)  # room=sid always - see module docstring
    finally:
        session.lock.release()


def _reap_idle_sessions():
    while True:
        time.sleep(IDLE_REAP_CHECK_PERIOD_S)
        now = time.monotonic()
        with _sessions_lock:
            stale = [sid for sid, s in _sessions.items() if now - s.last_activity > IDLE_REAP_SECONDS]
            for sid in stale:
                session = _sessions.pop(sid)
                session.landmarker.close()


def main(argv=None) -> int:
    global _enable_real, _udp_target, _udp_socket

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("GESTURE_MOCK_PORT", 5055)))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--enable-real", action="store_true",
                        help="Open a localhost UDP socket sending landmark data for teleop_node. "
                             "OFF by default - see module docstring's structural guarantee.")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5056)
    parser.add_argument("--https", action="store_true",
                        help="Serve over HTTPS with a self-signed cert (generated once, "
                             "reused after). Browsers only expose navigator.mediaDevices "
                             "(and therefore getUserMedia) on https:// or http://localhost - "
                             "reaching this page by LAN IP over plain http:// leaves the "
                             "camera API undefined, silently, with no permission prompt.")
    parser.add_argument("--tls-cert-dir",
                        default=os.environ.get("GESTURE_TLS_CERT_DIR", "/opt/openarm-gesture/tls"),
                        help="Where the self-signed cert/key pair lives (created if missing).")
    args = parser.parse_args(argv)

    _enable_real = args.enable_real
    if _enable_real:
        _udp_target = (args.udp_host, args.udp_port)
        _udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    reaper = threading.Thread(target=_reap_idle_sessions, daemon=True)
    reaper.start()

    ssl_context = None
    if args.https:
        pair = tls_cert.ensure_self_signed_cert(Path(args.tls_cert_dir))
        if pair is None:
            print("[gesture_teleop] --https requested but the self-signed cert could not be "
                  "prepared (is openssl on PATH? is {} writable?) - falling back to plain "
                  "HTTP. The browser's camera API will stay unavailable over LAN IP."
                  .format(args.tls_cert_dir), file=sys.stderr)
        else:
            ssl_context = pair
            print(f"[gesture_teleop] HTTPS enabled, self-signed cert at {args.tls_cert_dir} - "
                  "the browser will show a one-time \"not trusted\" warning for this LAN "
                  "device; click through it (Advanced -> Proceed) to reach the page.")

    socketio.run(app, host=args.host, port=args.port, debug=False,
                 allow_unsafe_werkzeug=True, ssl_context=ssl_context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
