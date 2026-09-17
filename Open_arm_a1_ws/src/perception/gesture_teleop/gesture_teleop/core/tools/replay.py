#!/usr/bin/env python3
"""Offline CLI: landmark JSON -> retargeted joint angles. No ROS, no camera,
no browser, no mediapipe - exercises core/frames.py + core/retarget.py
directly against a recorded/synthetic session, for Stage 2 verification
(see the plan's staged rollout).

Input JSON shape (a list of frames):
[
  {
    "timestamp_s": 0.0,
    "world_landmarks": {"11": [x,y,z], "12": [...], ...},   // MediaPipe indices as strings
    "world_visibility": {"11": 1.0, ...},                    // optional, default 1.0
    "world_presence": {"11": 1.0, ...},                      // optional, default 1.0
    "image_landmarks": {"0": [x,y,z], ...}                   // optional, only needed for facing_camera
  },
  ...
]

Usage:
    python -m gesture_teleop.core.tools.replay session.json [--side left] [--mirror-mode mirror]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from gesture_teleop.core import frames as fr
from gesture_teleop.core.pipeline import GestureSession
from gesture_teleop.core.types import LandmarkPoint, Skeleton


def _load_skeleton(frame: dict) -> Skeleton:
    world = {}
    for key, xyz in frame.get("world_landmarks", {}).items():
        idx = int(key)
        vis = frame.get("world_visibility", {}).get(key, 1.0)
        pres = frame.get("world_presence", {}).get(key, 1.0)
        world[idx] = LandmarkPoint(xyz=np.array(xyz, dtype=np.float64), visibility=vis, presence=pres)
    image = {}
    for key, xyz in frame.get("image_landmarks", {}).items():
        idx = int(key)
        image[idx] = LandmarkPoint(xyz=np.array(xyz, dtype=np.float64), visibility=1.0, presence=1.0)
    return Skeleton(timestamp_s=float(frame["timestamp_s"]), image_landmarks=image, world_landmarks=world)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_json", help="Path to a recorded/synthetic landmark session (see module docstring)")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--mirror-mode", choices=["mirror", "same_side"], default="mirror")
    parser.add_argument("--auto-engage", action="store_true",
                        help="Call engage() as soon as TRACKING is reached, for a quick smoke run")
    args = parser.parse_args(argv)

    with open(args.session_json, encoding="utf-8") as f:
        frames_data = json.load(f)

    session = GestureSession(side=args.side, mirror_mode=args.mirror_mode)
    engaged = False
    for i, frame in enumerate(frames_data):
        skeleton = _load_skeleton(frame)
        torso_frame, reason = fr.build_torso_frame(skeleton)
        result = session.process_frame(skeleton, torso_frame, skeleton.timestamp_s)

        if args.auto_engage and not engaged and session.state.value == "TRACKING":
            ok, why = session.engage(skeleton.timestamp_s)
            engaged = ok
            print(f"frame {i}: engage() -> {ok} {why}", file=sys.stderr)

        q_str = ", ".join(f"{v:+.4f}" for v in result.q)
        print(f"frame {i:4d} t={skeleton.timestamp_s:8.3f} state={session.state.value:10s} "
              f"valid={result.valid!s:5s} reason={result.reason.value:20s} q=[{q_str}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
