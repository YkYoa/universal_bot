"""The landmark datagram shared between mock_server.py (sender, only when
started with --enable-real) and teleop_node.py (receiver).

Plain JSON over UDP on localhost, not a ROS message: this keeps the
mock/mediapipe process and the ROS/actuation process in fully separate
Python environments (mock_server.py runs in the mediapipe venv;
teleop_node.py runs under the system ROS install - see the plan's
dependency-isolation section) with no shared library requirement beyond
the socket itself, and it means "the mockup cannot command the robot" is
structural rather than a matter of what one process chooses to send: this
module carries only raw landmark data, never a joint command, and
teleop_node.py is the only thing that ever runs retargeting on it for real
actuation.

UDP (not TCP) is deliberate: a landmark stream is a series of independent,
timestamped snapshots where the latest one matters and a dropped/stale
packet should never block the next - the same "latest wins" principle
copy_bridge's `:8090/frame/<slot>/latest` uses, applied to a push instead
of a pull.
"""
from __future__ import annotations

import json
import socket
from typing import Optional

import numpy as np

from gesture_teleop.core.types import LandmarkPoint, Skeleton

MAX_DATAGRAM_BYTES = 8192  # a few dozen landmarks as JSON comfortably fits well under this


def encode_skeleton(skeleton: Skeleton) -> bytes:
    payload = {
        "timestamp_s": skeleton.timestamp_s,
        "world_landmarks": {str(k): v.xyz.tolist() for k, v in skeleton.world_landmarks.items()},
        "world_visibility": {str(k): v.visibility for k, v in skeleton.world_landmarks.items()},
        "world_presence": {str(k): v.presence for k, v in skeleton.world_landmarks.items()},
        "image_landmarks": {str(k): v.xyz.tolist() for k, v in skeleton.image_landmarks.items()},
    }
    return json.dumps(payload).encode("utf-8")


def decode_skeleton(data: bytes) -> Optional[Skeleton]:
    try:
        payload = json.loads(data.decode("utf-8"))
        world = {
            int(k): LandmarkPoint(
                xyz=np.array(v, dtype=np.float64),
                visibility=payload.get("world_visibility", {}).get(k, 1.0),
                presence=payload.get("world_presence", {}).get(k, 1.0),
            )
            for k, v in payload["world_landmarks"].items()
        }
        image = {
            int(k): LandmarkPoint(xyz=np.array(v, dtype=np.float64), visibility=1.0, presence=1.0)
            for k, v in payload.get("image_landmarks", {}).items()
        }
        return Skeleton(timestamp_s=float(payload["timestamp_s"]), image_landmarks=image, world_landmarks=world)
    except (KeyError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None  # malformed packet - caller's watchdog handles the resulting staleness


def send_skeleton(sock: socket.socket, addr: tuple, skeleton: Skeleton) -> None:
    try:
        sock.sendto(encode_skeleton(skeleton), addr)
    except OSError:
        pass  # best-effort - the receiver's own watchdog (plan interlock 4) handles silence


def open_receiver(host: str, port: int, timeout_s: Optional[float] = None) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    if timeout_s is not None:
        sock.settimeout(timeout_s)
    return sock


def recv_skeleton(sock: socket.socket) -> Optional[Skeleton]:
    """One non-blocking-if-timeout-set receive. Returns None on timeout,
    malformed data, or any socket error - never raises, since a missing
    packet is exactly the normal "nothing new this tick" case for the
    caller's watchdog to handle, not an exceptional one."""
    try:
        data, _addr = sock.recvfrom(MAX_DATAGRAM_BYTES)
    except (socket.timeout, OSError):
        return None
    return decode_skeleton(data)
