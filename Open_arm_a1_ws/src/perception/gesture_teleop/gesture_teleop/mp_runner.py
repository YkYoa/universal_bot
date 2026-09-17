"""The ONLY module in this package that imports mediapipe (enforced by
test_no_ros_in_core.py for core/, and this module lives outside core/
specifically because of that import). Wraps MediaPipe Pose Landmarker
(Tasks API, RunningMode.VIDEO) behind a small interface that converts
straight into core.types.Skeleton, and provides a MP_BACKEND=fake
in-process replay backend so mock_server.py's integration tests
(test_mock_server.py) never need a real mediapipe install or a camera.

RunningMode.VIDEO (not IMAGE): reuses the previous frame's ROI internally
and is dramatically more stable than re-detecting from scratch every frame
- this buys more real smoothing than the One Euro filter downstream ever
could. It requires strictly increasing timestamps per instance (a
non-increasing one throws), so every instance here is fed a private
monotonic counter, never the browser's own clock (a client's clock can
jump, go backwards on tab switch, or simply lie).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gesture_teleop.core.landmarks import REQUIRED_LANDMARK_INDICES
from gesture_teleop.core.types import LandmarkPoint, Skeleton

_DEFAULT_MODEL_PATH = (
    "/opt/robot-healthmate-agent-os/current/ai-services/vision-api/models/pose_landmarker_lite.task"
)


class PoseLandmarkerUnavailable(RuntimeError):
    """Raised when MP_BACKEND=real is requested but mediapipe cannot be
    imported or the model file cannot be found - never silently falls back
    to the fake backend, since that would be indistinguishable from a real
    (but empty) detection to a caller."""


class LandmarkerBase:
    def detect(self, frame_bgr: np.ndarray, wall_clock_s: float) -> Optional[Skeleton]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class RealPoseLandmarker(LandmarkerBase):
    """One instance per session (see mock_server.py) - PoseLandmarker is
    NOT thread-safe and costs ~150-200ms to construct, so it is created
    once on connect and closed on disconnect, never shared across
    sessions."""

    def __init__(self, model_path: Optional[str] = None):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise PoseLandmarkerUnavailable(f"mediapipe not importable: {exc}") from exc

        path = model_path or os.environ.get("GESTURE_POSE_MODEL_PATH", _DEFAULT_MODEL_PATH)
        if not os.path.isfile(path):
            raise PoseLandmarkerUnavailable(f"pose landmarker model not found at {path}")

        self._mp = mp
        base_options = mp_python.BaseOptions(model_asset_path=path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr: np.ndarray, wall_clock_s: float) -> Optional[Skeleton]:
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)

        # Server's own monotonic clock, never the client's - see module
        # docstring. Timestamps must be strictly increasing for this
        # instance; guard against two calls landing in the same
        # millisecond (possible on a fast loopback test).
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return _to_skeleton(result, wall_clock_s)

    def close(self) -> None:
        self._landmarker.close()


def _to_skeleton(mp_result, wall_clock_s: float) -> Optional[Skeleton]:
    if not mp_result.pose_landmarks or not mp_result.pose_world_landmarks:
        return None
    image_lms = mp_result.pose_landmarks[0]
    world_lms = mp_result.pose_world_landmarks[0]

    image_points = {}
    world_points = {}
    for idx in REQUIRED_LANDMARK_INDICES:
        if idx >= len(image_lms) or idx >= len(world_lms):
            continue
        il = image_lms[idx]
        wl = world_lms[idx]
        vis = getattr(il, "visibility", 1.0)
        pres = getattr(il, "presence", 1.0)
        image_points[idx] = LandmarkPoint(xyz=np.array([il.x, il.y, il.z]), visibility=vis, presence=pres)
        world_points[idx] = LandmarkPoint(xyz=np.array([wl.x, wl.y, wl.z]), visibility=vis, presence=pres)

    return Skeleton(timestamp_s=wall_clock_s, image_landmarks=image_points, world_landmarks=world_points)


@dataclass
class FakePoseLandmarker(LandmarkerBase):
    """Replays a pre-built sequence of Skeleton (or None, for "no person
    detected this frame") objects, ignoring the actual frame bytes -
    MP_BACKEND=fake, used by test_mock_server.py so the WebSocket/session
    integration tests need neither a real mediapipe install nor a camera.
    """

    skeletons: list = field(default_factory=list)
    _index: int = field(default=0, init=False)

    def detect(self, frame_bgr: np.ndarray, wall_clock_s: float) -> Optional[Skeleton]:
        if not self.skeletons:
            return None
        skeleton = self.skeletons[self._index % len(self.skeletons)]
        self._index += 1
        if skeleton is None:
            return None
        # Re-stamp with the wall clock actually passed in, so staleness
        # checks downstream see a sensible timestamp regardless of how the
        # fixture was authored.
        return Skeleton(timestamp_s=wall_clock_s, image_landmarks=skeleton.image_landmarks,
                         world_landmarks=skeleton.world_landmarks)


def create_landmarker(model_path: Optional[str] = None) -> LandmarkerBase:
    """Factory respecting MP_BACKEND (default "real"). "fake" returns an
    EMPTY FakePoseLandmarker (no detections ever) unless the caller
    constructs+injects their own FakePoseLandmarker(skeletons=[...])
    directly - this factory exists for mock_server.py's normal startup
    path, tests build FakePoseLandmarker themselves."""
    backend = os.environ.get("MP_BACKEND", "real")
    if backend == "fake":
        return FakePoseLandmarker(skeletons=[])
    return RealPoseLandmarker(model_path)
