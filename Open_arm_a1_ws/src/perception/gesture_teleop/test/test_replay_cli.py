"""Smoke test for core/tools/replay.py - the offline "landmark JSON in,
joint angles out" CLI with no ROS/camera/browser/mediapipe (plan Stage 2
verification: `python -m gesture_teleop.core.tools.replay session.json`).
"""
from pathlib import Path

from gesture_teleop.core.tools import replay

FIXTURE = Path(__file__).parent / "fixtures" / "session.json"


def test_replay_runs_clean_and_reaches_engaged(capsys):
    assert FIXTURE.is_file()
    rc = replay.main([str(FIXTURE), "--side", "left", "--mirror-mode", "mirror", "--auto-engage"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "state=ENGAGED" in out
    assert "valid=True" in out


def test_replay_same_side_mode_also_runs(capsys):
    rc = replay.main([str(FIXTURE), "--side", "right", "--mirror-mode", "same_side"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "frame    0" in out
