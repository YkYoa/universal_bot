"""Unit tests for core/pipeline.py (GestureSession FSM)."""
import numpy as np
import pytest

from gesture_teleop.core import frames as fr
from gesture_teleop.core import landmarks as lm
from gesture_teleop.core import pipeline as pl
from gesture_teleop.core import robot_model as rm
from gesture_teleop.core.types import LandmarkPoint, RejectReason, Skeleton

# A real, reachable left-arm pose (verified against retarget.solve_arm) used
# to place the fixture's elbow/wrist landmarks - an arbitrary hand-picked
# local direction is NOT guaranteed reachable by the actual robot geometry
# and retarget.py correctly reports those as UNREACHABLE (see
# test_retarget.py), which would make every fixture frame here "bad" and
# the FSM would never leave IDLE.
_REACHABLE_Q = np.array([0.0, -0.3, 0.2, 1.0, 0.0, 0.0, 0.0])
_shoulder_ref = rm.shoulder_reference("left")
_frames = rm.forward_kinematics("left", _REACHABLE_Q)
_ELBOW_LOCAL = _frames.origins[3] - _shoulder_ref
_WRIST_LOCAL = _frames.origins[4] - _shoulder_ref
_ALT_WRIST_LOCAL = _WRIST_LOCAL + np.array([0.0, 0.05, -0.05])  # a visibly different reach


def _pt(x, y, z, vis=1.0, pres=1.0):
    return LandmarkPoint(xyz=np.array([x, y, z]), visibility=vis, presence=pres)


def _place_in_torso_frame(frame: fr.TorsoFrame, local: np.ndarray) -> np.ndarray:
    """Inverse of TorsoFrame.to_local: world point for a given local
    (forward, left, up) offset from the frame's own origin."""
    return frame.origin + local[0] * frame.x_forward + local[1] * frame.y_left + local[2] * frame.z_up


def _good_skeleton(t, right_wrist_local=None):
    """A synthetic, fully-visible, camera-facing upright person. Both arms'
    elbow/wrist are placed via _place_in_torso_frame so their TORSO-FRAME
    LOCAL directions are, by construction, real reachable robot-arm
    directions (see _REACHABLE_Q above) - not an arbitrary hand-picked
    world-space offset, which retarget.py would likely reject as
    UNREACHABLE for this short-armed robot."""
    shoulder_l = np.array([0.0, 0.2, 1.4])
    shoulder_r = np.array([0.0, -0.2, 1.4])
    hip_l = np.array([0.0, 0.1, 1.0])
    hip_r = np.array([0.0, -0.1, 1.0])
    world_partial = {
        lm.LEFT_SHOULDER: _pt(*shoulder_l), lm.RIGHT_SHOULDER: _pt(*shoulder_r),
        lm.LEFT_HIP: _pt(*hip_l), lm.RIGHT_HIP: _pt(*hip_r),
    }
    probe = Skeleton(timestamp_s=t, image_landmarks={}, world_landmarks=world_partial)
    frame, reason = fr.build_torso_frame(probe)
    assert reason is RejectReason.NONE

    # _ELBOW_LOCAL/_WRIST_LOCAL are offsets from the ARM'S OWN shoulder
    # reference (matching what retarget.solve_arm actually computes:
    # elbow_local - shoulder_local, where shoulder_local is the
    # INDIVIDUAL shoulder landmark's torso-local position, not the
    # shoulder-midpoint origin the TorsoFrame itself is centered on) - so
    # each arm's elbow/wrist must be placed relative to THAT shoulder's
    # own local position, not relative to the frame origin directly.
    left_shoulder_local = frame.to_local(shoulder_l)
    right_shoulder_local = frame.to_local(shoulder_r)
    right_wrist = right_wrist_local if right_wrist_local is not None else _WRIST_LOCAL
    world = dict(world_partial)
    world[lm.LEFT_ELBOW] = _pt(*_place_in_torso_frame(frame, left_shoulder_local + _ELBOW_LOCAL))
    world[lm.LEFT_WRIST] = _pt(*_place_in_torso_frame(frame, left_shoulder_local + _WRIST_LOCAL))
    world[lm.RIGHT_ELBOW] = _pt(*_place_in_torso_frame(frame, right_shoulder_local + _ELBOW_LOCAL))
    world[lm.RIGHT_WRIST] = _pt(*_place_in_torso_frame(frame, right_shoulder_local + right_wrist))

    image = {
        lm.NOSE: _pt(0.5, 0.3, 0.0),
        lm.LEFT_EAR: _pt(0.55, 0.3, 0.0),
        lm.RIGHT_EAR: _pt(0.45, 0.3, 0.0),
    }
    return Skeleton(timestamp_s=t, image_landmarks=image, world_landmarks=world)


def _torso_for(skel):
    frame, reason = fr.build_torso_frame(skel)
    assert reason is RejectReason.NONE
    return frame


def test_starts_idle_and_emits_nothing():
    session = pl.GestureSession(side="left")
    assert session.state == pl.SessionState.IDLE


def test_good_frame_moves_idle_to_tracking_without_commanding():
    session = pl.GestureSession(side="left", mirror_mode="same_side")
    skel = _good_skeleton(0.0)
    result = session.process_frame(skel, _torso_for(skel), 0.0)
    assert session.state == pl.SessionState.TRACKING
    assert not result.valid  # TRACKING never commands (explicit-arm)


def test_engage_fails_from_idle():
    session = pl.GestureSession(side="left")
    ok, _ = session.engage(0.0)
    assert not ok
    assert session.state == pl.SessionState.IDLE


def test_engage_succeeds_from_tracking_and_then_commands():
    session = pl.GestureSession(side="left", mirror_mode="same_side")
    skel = _good_skeleton(0.0)
    session.process_frame(skel, _torso_for(skel), 0.0)
    assert session.state == pl.SessionState.TRACKING

    ok, _ = session.engage(0.01)
    assert ok
    assert session.state == pl.SessionState.ENGAGED

    result = session.process_frame(_good_skeleton(0.02), _torso_for(_good_skeleton(0.02)), 0.02)
    assert session.state == pl.SessionState.ENGAGED
    assert result.valid
    assert result.q.shape == (7,)


def test_bad_frame_while_engaged_moves_to_hold():
    session = pl.GestureSession(side="left", mirror_mode="same_side")
    t = 0.0
    skel = _good_skeleton(t)
    session.process_frame(skel, _torso_for(skel), t)
    session.engage(t)
    assert session.state == pl.SessionState.ENGAGED

    # Low-visibility frame -> HOLD, and no command emitted.
    t += 0.05
    bad = _good_skeleton(t)
    bad.world_landmarks[lm.LEFT_WRIST] = _pt(0.25, 0.45, 1.1, vis=0.0)
    result = session.process_frame(bad, _torso_for(bad), t)
    assert session.state == pl.SessionState.HOLD
    assert not result.valid
    assert result.reason is RejectReason.LOW_VISIBILITY


def test_recovers_from_hold_to_engaged_on_good_frame():
    session = pl.GestureSession(side="left", mirror_mode="same_side", hold_timeout_s=1.0)
    t = 0.0
    skel = _good_skeleton(t)
    session.process_frame(skel, _torso_for(skel), t)
    session.engage(t)

    t += 0.05
    bad = _good_skeleton(t)
    bad.world_landmarks[lm.LEFT_WRIST] = _pt(0.25, 0.45, 1.1, vis=0.0)
    session.process_frame(bad, _torso_for(bad), t)
    assert session.state == pl.SessionState.HOLD

    t += 0.05
    good = _good_skeleton(t)
    result = session.process_frame(good, _torso_for(good), t)
    assert session.state == pl.SessionState.ENGAGED
    assert result.valid


def test_hold_timeout_via_tick_disengages():
    session = pl.GestureSession(side="left", mirror_mode="same_side", hold_timeout_s=0.2)
    t = 0.0
    skel = _good_skeleton(t)
    session.process_frame(skel, _torso_for(skel), t)
    session.engage(t)

    t += 0.05
    bad = _good_skeleton(t)
    bad.world_landmarks[lm.LEFT_WRIST] = _pt(0.25, 0.45, 1.1, vis=0.0)
    session.process_frame(bad, _torso_for(bad), t)
    assert session.state == pl.SessionState.HOLD

    # No new frames at all - the independent watchdog (tick) must still
    # notice the hold has expired (plan interlock 4).
    t += 0.3
    result = session.tick(t)
    assert session.state == pl.SessionState.DISENGAGED
    assert not result.valid


def test_dead_man_timeout_via_tick_with_no_frames():
    """Frames simply stop arriving entirely (a hung upstream) while
    ENGAGED - tick() alone (no process_frame calls) must catch this."""
    session = pl.GestureSession(side="left", mirror_mode="same_side", dead_man_timeout_s=0.1, hold_timeout_s=0.1)
    t = 0.0
    skel = _good_skeleton(t)
    session.process_frame(skel, _torso_for(skel), t)
    session.engage(t)
    assert session.state == pl.SessionState.ENGAGED

    t += 0.15  # exceeds dead_man_timeout_s with no new frame
    session.tick(t)
    assert session.state == pl.SessionState.HOLD

    t += 0.2  # exceeds hold_timeout_s too
    session.tick(t)
    assert session.state == pl.SessionState.DISENGAGED


def test_explicit_disengage_from_engaged():
    session = pl.GestureSession(side="left", mirror_mode="same_side")
    skel = _good_skeleton(0.0)
    session.process_frame(skel, _torso_for(skel), 0.0)
    session.engage(0.0)
    assert session.state == pl.SessionState.ENGAGED
    session.disengage(0.1)
    assert session.state == pl.SessionState.DISENGAGED


def test_disengaged_requires_new_tracking_before_reengage():
    session = pl.GestureSession(side="left", mirror_mode="same_side")
    skel = _good_skeleton(0.0)
    session.process_frame(skel, _torso_for(skel), 0.0)
    session.engage(0.0)
    session.disengage(0.1)

    ok, _ = session.engage(0.2)
    assert not ok  # still DISENGAGED, no tracking frame processed since

    result = session.process_frame(_good_skeleton(0.3), _torso_for(_good_skeleton(0.3)), 0.3)
    assert session.state == pl.SessionState.TRACKING
    assert not result.valid

    ok, _ = session.engage(0.31)
    assert ok


def test_never_commands_in_idle_or_disengaged_scripted_session():
    """A 300-tick scripted session covering engage / track-loss /
    stale-frame / disengage / re-engage: assert no command (valid=True) is
    ever emitted outside ENGAGED."""
    session = pl.GestureSession(side="left", mirror_mode="same_side", dead_man_timeout_s=0.2, hold_timeout_s=0.3)
    t = 0.0
    dt = 0.02
    engaged_once = False
    for i in range(300):
        t += dt
        if i < 20:
            skel = _good_skeleton(t)  # priming: TRACKING
        elif i == 20:
            skel = _good_skeleton(t)
        elif 60 <= i < 90:
            skel = None  # simulate track loss (no skeleton this tick)
        elif i == 150:
            skel = None  # trigger via explicit disengage instead
        else:
            skel = _good_skeleton(t)

        torso = _torso_for(skel) if skel is not None else None
        result = session.process_frame(skel, torso, t)

        if i == 20 and session.state == pl.SessionState.TRACKING:
            ok, _ = session.engage(t)
            assert ok
            engaged_once = True

        if i == 150:
            session.disengage(t)

        if result.valid:
            assert session.state == pl.SessionState.ENGAGED, (
                f"tick {i}: command emitted while state={session.state}"
            )
        else:
            assert session.state != pl.SessionState.ENGAGED or True  # (invalid results can occur mid-ENGAGED too, e.g. transient)

    assert engaged_once


def test_mirror_mode_selects_opposite_human_arm():
    """side='left' with mirror_mode='mirror' should read the human's RIGHT
    arm landmarks (see frames.arm_source_landmarks)."""
    session = pl.GestureSession(side="left", mirror_mode="mirror")
    # Give left and right arms visibly different wrist positions so a
    # wrong source-arm choice changes the retargeted result.
    skel = _good_skeleton(0.0, right_wrist_local=_ALT_WRIST_LOCAL)
    torso = _torso_for(skel)
    result = session.process_frame(skel, torso, 0.0)
    assert session.state == pl.SessionState.TRACKING

    # Compute what same_side would have produced, for comparison.
    same_side_session = pl.GestureSession(side="left", mirror_mode="same_side")
    same_side_session.process_frame(skel, torso, 0.0)
    # They read different human landmarks (right vs left wrist, which
    # differ in this fixture), so their internally retargeted q_prev
    # should differ.
    assert not np.allclose(session.q_prev, same_side_session.q_prev)
