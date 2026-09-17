"""Unit tests for udp_link.py's encode/decode - the wire format shared
between mock_server.py and teleop_node.py."""
import socket

import numpy as np
import pytest

from gesture_teleop import udp_link
from gesture_teleop.core.types import LandmarkPoint, Skeleton


def _sample_skeleton():
    world = {11: LandmarkPoint(xyz=np.array([0.1, 0.2, 0.3]), visibility=0.9, presence=0.8)}
    image = {0: LandmarkPoint(xyz=np.array([0.5, 0.5, 0.0]), visibility=1.0, presence=1.0)}
    return Skeleton(timestamp_s=12.5, image_landmarks=image, world_landmarks=world)


def test_encode_decode_round_trip():
    skel = _sample_skeleton()
    data = udp_link.encode_skeleton(skel)
    decoded = udp_link.decode_skeleton(data)
    assert decoded is not None
    assert decoded.timestamp_s == pytest.approx(12.5)
    np.testing.assert_allclose(decoded.world_landmarks[11].xyz, [0.1, 0.2, 0.3])
    assert decoded.world_landmarks[11].visibility == pytest.approx(0.9)
    assert decoded.world_landmarks[11].presence == pytest.approx(0.8)
    np.testing.assert_allclose(decoded.image_landmarks[0].xyz, [0.5, 0.5, 0.0])


def test_decode_malformed_data_returns_none():
    assert udp_link.decode_skeleton(b"not json at all") is None
    assert udp_link.decode_skeleton(b'{"missing": "fields"}') is None


def test_encoded_payload_fits_max_datagram():
    data = udp_link.encode_skeleton(_sample_skeleton())
    assert len(data) < udp_link.MAX_DATAGRAM_BYTES


def test_send_and_receive_over_real_localhost_socket():
    receiver = udp_link.open_receiver("127.0.0.1", 0, timeout_s=1.0)
    port = receiver.getsockname()[1]
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        udp_link.send_skeleton(sender, ("127.0.0.1", port), _sample_skeleton())
        received = udp_link.recv_skeleton(receiver)
        assert received is not None
        np.testing.assert_allclose(received.world_landmarks[11].xyz, [0.1, 0.2, 0.3])
    finally:
        sender.close()
        receiver.close()


def test_recv_skeleton_times_out_gracefully():
    receiver = udp_link.open_receiver("127.0.0.1", 0, timeout_s=0.05)
    try:
        assert udp_link.recv_skeleton(receiver) is None
    finally:
        receiver.close()
