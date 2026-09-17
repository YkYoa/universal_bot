"""Architectural invariant: mock_server.py (and everything under web/) must
never be able to command the real robot - the structural proof behind
"nothing real, only mockup" (see mock_server.py's module docstring).
"""
import ast
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "gesture_teleop"
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MOCK_SERVER = PACKAGE_DIR / "mock_server.py"

FORBIDDEN_IMPORTS = {"rclpy", "command_sink", "controller_switch"}
FORBIDDEN_STRINGS = ("forward_position_controller", "arm_controller/joint_trajectory")


def _imported_modules(tree: ast.AST):
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_mock_server_imports_no_ros_or_command_modules():
    tree = ast.parse(MOCK_SERVER.read_text(encoding="utf-8"))
    imported = _imported_modules(tree)
    hit = imported & FORBIDDEN_IMPORTS
    assert not hit, f"mock_server.py imports forbidden module(s): {hit}"


def test_mock_server_never_mentions_the_real_command_topic():
    text = MOCK_SERVER.read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_STRINGS:
        assert forbidden not in text, f"mock_server.py mentions {forbidden!r} - it must never address the real command interface"


def test_web_assets_never_mention_the_real_command_topic():
    if not WEB_DIR.is_dir():
        return
    for path in WEB_DIR.rglob("*"):
        if path.is_dir() or "vendor" in path.parts:
            continue  # third-party vendored files (socket.io client) are out of scope
        text = path.read_text(encoding="utf-8", errors="ignore")
        for forbidden in FORBIDDEN_STRINGS:
            assert forbidden not in text, f"{path} mentions {forbidden!r}"


def test_mock_server_udp_bridge_is_off_by_default():
    """The only bridge toward the real robot is the optional UDP socket,
    and it must default to OFF (--enable-real not passed)."""
    tree = ast.parse(MOCK_SERVER.read_text(encoding="utf-8"))
    found_default_false = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            args = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if "--enable-real" in args:
                for kw in node.keywords:
                    if kw.arg == "action" and isinstance(kw.value, ast.Constant) and kw.value.value == "store_true":
                        found_default_false = True
    assert found_default_false, "--enable-real must be an action='store_true' flag (default False)"
