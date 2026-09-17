"""Unit tests for command_sink.py - all sinks are exercised with injected
plain callables, no ROS."""
import numpy as np
import pytest

from gesture_teleop import command_sink as cs


def test_null_sink_does_nothing():
    sink = cs.NullSink()
    sink.publish("left", np.zeros(7))  # must not raise


def test_recording_sink_records_calls():
    sink = cs.RecordingSink()
    sink.publish("left", np.arange(7, dtype=float))
    sink.publish("right", np.ones(7))
    assert len(sink.calls) == 2
    assert sink.calls[0][0] == "left"
    np.testing.assert_allclose(sink.calls[0][1], np.arange(7, dtype=float))
    assert sink.calls[1][0] == "right"


def test_recording_sink_copies_not_aliases():
    sink = cs.RecordingSink()
    q = np.zeros(7)
    sink.publish("left", q)
    q[0] = 99.0
    assert sink.calls[0][1][0] == 0.0  # unaffected by later mutation of the caller's array


def test_forward_position_sink_routes_by_side():
    left_calls = []
    right_calls = []
    sink = cs.ForwardPositionSink(
        publish_left=left_calls.append,
        publish_right=right_calls.append,
        make_msg=lambda data: {"data": data},
    )
    sink.publish("left", np.arange(7, dtype=float))
    sink.publish("right", np.ones(7))
    assert len(left_calls) == 1 and len(right_calls) == 1
    assert left_calls[0]["data"] == list(range(7))
    assert right_calls[0]["data"] == [1.0] * 7


def test_forward_position_sink_rejects_wrong_shape():
    sink = cs.ForwardPositionSink(publish_left=lambda m: None, publish_right=lambda m: None,
                                   make_msg=lambda d: d)
    with pytest.raises(ValueError):
        sink.publish("left", np.zeros(5))


def test_jtc_sink_computes_velocity_from_previous_q():
    msgs = []
    sink = cs.JtcSink(
        publish_left=msgs.append, publish_right=lambda m: None,
        make_msg=lambda pos, vel: {"pos": pos, "vel": vel},
        step_s=0.1,
    )
    sink.publish("left", np.zeros(7))
    sink.publish("left", np.full(7, 0.1))
    assert msgs[0]["vel"] == [0.0] * 7  # first call: no previous q, velocity is 0
    np.testing.assert_allclose(msgs[1]["vel"], [1.0] * 7, atol=1e-9)  # (0.1-0.0)/0.1


def test_jtc_sink_tracks_left_right_independently():
    sink = cs.JtcSink(publish_left=lambda m: None, publish_right=lambda m: None,
                       make_msg=lambda pos, vel: (pos, vel), step_s=0.1)
    sink.publish("left", np.full(7, 1.0))
    sink.publish("right", np.full(7, 5.0))
    assert sink._prev_q["left"][0] == 1.0
    assert sink._prev_q["right"][0] == 5.0


def test_preview_sink_uses_side_specific_joint_names():
    msgs = []
    sink = cs.PreviewSink(publish_fn=msgs.append, make_msg=lambda names, pos: {"name": names, "position": pos})
    sink.publish("left", np.zeros(7))
    assert msgs[0]["name"] == cs.JOINT_NAMES["left"]
    sink.publish("right", np.zeros(7))
    assert msgs[1]["name"] == cs.JOINT_NAMES["right"]


def test_joint_names_have_seven_entries_each_side():
    assert len(cs.JOINT_NAMES["left"]) == 7
    assert len(cs.JOINT_NAMES["right"]) == 7
    assert cs.JOINT_NAMES["left"][0] == "openarm_left_joint1"
    assert cs.JOINT_NAMES["right"][6] == "openarm_right_joint7"
