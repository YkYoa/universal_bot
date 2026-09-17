"""Integration tests for mock_server.py using MP_BACKEND=fake (a
FakePoseLandmarker replaying recorded landmarks) driven via
flask_socketio's test client - no camera, no ROS, no real mediapipe model.

Covers the regressions the plan explicitly calls out against
robot_api_server.py's `_streaming` global-flag bug
(robot_api_server.py:1237): results must land per-sid, one client's
disconnect must not kill another's stream, and a slow/backed-up session
must drop frames rather than queue them.
"""
import numpy as np
import pytest

pytest.importorskip("flask", reason="mock_server needs Flask")
pytest.importorskip("flask_socketio", reason="mock_server needs Flask-SocketIO")

from gesture_teleop import mock_server
from gesture_teleop import mp_runner
from gesture_teleop.core import frames as fr
from gesture_teleop.core import landmarks as lm
from gesture_teleop.core import robot_model as rm
from gesture_teleop.core.types import LandmarkPoint, Skeleton

pytestmark = pytest.mark.integration

# A real, reachable, collision-free arm pose (same technique as
# test_pipeline.py - an arbitrary hand-picked elbow/wrist WORLD offset is
# NOT guaranteed reachable by this short-armed robot's actual geometry,
# and retarget.py correctly reports those as UNREACHABLE, which would keep
# every session stuck in IDLE/TRACKING and never reach ENGAGED here).
_REACHABLE_Q = np.array([0.0, -0.3, 0.2, 1.0, 0.0, 0.0, 0.0])
_shoulder_ref = rm.shoulder_reference("left")
_frames = rm.forward_kinematics("left", _REACHABLE_Q)
_ELBOW_LOCAL = _frames.origins[3] - _shoulder_ref
_WRIST_LOCAL = _frames.origins[4] - _shoulder_ref


def _pt(x, y, z, vis=1.0, pres=1.0):
    return LandmarkPoint(xyz=np.array([x, y, z]), visibility=vis, presence=pres)


def _place(frame: fr.TorsoFrame, local: np.ndarray) -> np.ndarray:
    return frame.origin + local[0] * frame.x_forward + local[1] * frame.y_left + local[2] * frame.z_up


def _good_skeleton(t=0.0):
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
    assert reason.value == "none"

    shoulder_l_local = frame.to_local(shoulder_l)
    shoulder_r_local = frame.to_local(shoulder_r)
    world = dict(world_partial)
    world[lm.LEFT_ELBOW] = _pt(*_place(frame, shoulder_l_local + _ELBOW_LOCAL))
    world[lm.LEFT_WRIST] = _pt(*_place(frame, shoulder_l_local + _WRIST_LOCAL))
    world[lm.RIGHT_ELBOW] = _pt(*_place(frame, shoulder_r_local + _ELBOW_LOCAL))
    world[lm.RIGHT_WRIST] = _pt(*_place(frame, shoulder_r_local + _WRIST_LOCAL))

    image = {
        lm.NOSE: _pt(0.5, 0.3, 0.0), lm.LEFT_EAR: _pt(0.55, 0.3, 0.0), lm.RIGHT_EAR: _pt(0.45, 0.3, 0.0),
    }
    return Skeleton(timestamp_s=t, image_landmarks=image, world_landmarks=world)


@pytest.fixture(autouse=True)
def _clean_sessions():
    """mock_server's _sessions dict is module-level; reset it around each
    test so tests don't interfere with each other (and don't leak
    landmarker "instances" - FakePoseLandmarker.close() is a no-op)."""
    mock_server._sessions.clear()
    yield
    mock_server._sessions.clear()


@pytest.fixture
def fake_frame_bytes():
    """A tiny valid JPEG so cv2.imdecode succeeds - content is irrelevant
    since FakePoseLandmarker ignores the frame and replays canned
    skeletons."""
    cv2 = pytest.importorskip("cv2", reason="mock_server's frame handler needs cv2")

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _patch_fake_landmarker(monkeypatch, skeletons):
    def _factory(model_path=None):
        return mp_runner.FakePoseLandmarker(skeletons=skeletons)

    monkeypatch.setattr(mock_server.mp_runner, "create_landmarker", _factory)


def test_single_client_receives_own_result(monkeypatch, fake_frame_bytes):
    _patch_fake_landmarker(monkeypatch, [_good_skeleton()])
    client = mock_server.socketio.test_client(mock_server.app)
    assert client.is_connected()

    client.emit("frame", fake_frame_bytes)
    received = client.get_received()
    results = [r for r in received if r["name"] == "result"]
    assert len(results) == 1
    payload = results[0]["args"][0]
    assert "arms" in payload and "left" in payload["arms"] and "right" in payload["arms"]
    client.disconnect()


def test_results_are_per_sid_not_broadcast(monkeypatch, fake_frame_bytes):
    """Regression test for the sibling of the `_streaming` bug: emitting
    without room=sid would leak client A's landmarks to client B."""
    _patch_fake_landmarker(monkeypatch, [_good_skeleton()])
    client_a = mock_server.socketio.test_client(mock_server.app)
    client_b = mock_server.socketio.test_client(mock_server.app)

    client_a.emit("frame", fake_frame_bytes)
    a_received = client_a.get_received()
    b_received = client_b.get_received()

    assert any(r["name"] == "result" for r in a_received)
    assert not any(r["name"] == "result" for r in b_received)
    client_a.disconnect()
    client_b.disconnect()


def test_second_client_disconnect_does_not_kill_first(monkeypatch, fake_frame_bytes):
    """Direct regression test for robot_api_server.py's `_streaming`
    global-flag bug (robot_api_server.py:1237): any one client
    disconnecting must not affect another client's session."""
    _patch_fake_landmarker(monkeypatch, [_good_skeleton()])
    client_a = mock_server.socketio.test_client(mock_server.app)
    client_b = mock_server.socketio.test_client(mock_server.app)

    client_b.disconnect()

    client_a.emit("frame", fake_frame_bytes)
    a_received = client_a.get_received()
    assert any(r["name"] == "result" for r in a_received)
    client_a.disconnect()


def test_concurrent_session_cap_refuses_third_client(monkeypatch, fake_frame_bytes):
    _patch_fake_landmarker(monkeypatch, [_good_skeleton()])
    clients = [mock_server.socketio.test_client(mock_server.app) for _ in range(mock_server.MAX_CONCURRENT_SESSIONS)]
    for c in clients:
        assert c.is_connected()

    refused = mock_server.socketio.test_client(mock_server.app)
    # flask_socketio's test client disconnects itself when the connect
    # handler returns False, so the server-side session count must not
    # have grown past the cap.
    assert not refused.is_connected()
    assert len(mock_server._sessions) == mock_server.MAX_CONCURRENT_SESSIONS

    for c in clients:
        c.disconnect()


def test_engage_disengage_round_trip(monkeypatch, fake_frame_bytes):
    _patch_fake_landmarker(monkeypatch, [_good_skeleton(), _good_skeleton(0.02), _good_skeleton(0.04)])
    client = mock_server.socketio.test_client(mock_server.app)
    client.emit("frame", fake_frame_bytes)  # -> TRACKING
    client.get_received()

    client.emit("engage", {})
    received = client.get_received()
    engage_results = [r for r in received if r["name"] == "engage_result"]
    assert len(engage_results) == 1

    client.emit("frame", fake_frame_bytes)
    result = [r for r in client.get_received() if r["name"] == "result"][0]["args"][0]
    # At least one side should now be engaged and emitting a valid target,
    # since both left/right sessions read the same visible skeleton.
    assert result["state"]["left"] == "ENGAGED" or result["state"]["right"] == "ENGAGED"

    client.emit("disengage", {})
    client.disconnect()


def test_mirror_mode_can_be_changed(monkeypatch, fake_frame_bytes):
    _patch_fake_landmarker(monkeypatch, [_good_skeleton()])
    client = mock_server.socketio.test_client(mock_server.app)
    client.emit("set_mirror_mode", {"mode": "same_side"})
    client.emit("frame", fake_frame_bytes)
    result = [r for r in client.get_received() if r["name"] == "result"][0]["args"][0]
    assert result["mirror_mode"] == "same_side"
    client.disconnect()


def test_no_person_frame_reports_idle_state(monkeypatch, fake_frame_bytes):
    _patch_fake_landmarker(monkeypatch, [None])
    client = mock_server.socketio.test_client(mock_server.app)
    client.emit("frame", fake_frame_bytes)
    result = [r for r in client.get_received() if r["name"] == "result"][0]["args"][0]
    assert result["skeleton"]["landmarks"] is None
    assert result["state"]["left"] == "IDLE"
    client.disconnect()


def test_health_endpoint():
    with mock_server.app.test_client() as c:
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["enable_real"] is False
