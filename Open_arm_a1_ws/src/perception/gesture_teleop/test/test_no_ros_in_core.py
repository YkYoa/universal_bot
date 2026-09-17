"""Architectural invariant: core/ must never import rclpy, flask,
mediapipe, or cv2. This is what makes the entire retargeting algorithm
testable on any machine with numpy and runnable identically from both the
mock server and the real teleop node - see the plan's "pure core" design
principle.
"""
import ast
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parent.parent / "gesture_teleop" / "core"
FORBIDDEN_MODULES = {"rclpy", "flask", "flask_socketio", "mediapipe", "cv2"}


def _core_python_files():
    return sorted(p for p in CORE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_modules(tree: ast.AST):
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


@pytest.mark.parametrize("path", _core_python_files(), ids=lambda p: str(p.relative_to(CORE_DIR)))
def test_core_file_imports_no_forbidden_module(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = _imported_modules(tree)
    forbidden_hit = imported & FORBIDDEN_MODULES
    assert not forbidden_hit, f"{path} imports forbidden module(s): {forbidden_hit}"


def test_core_directory_is_not_empty():
    """Guards against the parametrize above silently collecting zero cases
    if CORE_DIR's path ever changes shape."""
    assert len(_core_python_files()) >= 5
