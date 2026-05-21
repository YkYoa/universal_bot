#!/usr/bin/env python3
"""
sequence_to_bt.py
─────────────────
Convert an OpenArm sequence.yaml into a BehaviorTree.CPP v4 XML file.

Key-rules (mirrors the YAML header convention):
  Prefix:
    la  → left_arm     ra  → right_arm
    lh  → left_gripper rh  → right_gripper
    head → (skipped — head is not a MoveIt planning group in this system)

  Goal type (determined by key name):
    contains "Angle"  → joint-space  → PlanToNamedPose (joints resolved from SRDF)
                         NOTE: angles stored in YAML; the skill server looks them
                         up by named_pose from the SRDF.  If you want raw joint
                         angles from YAML, extend SkillRequest with joint_targets.
    contains "Joint"  → Cartesian pose  → PlanToPose (x,y,z,qx,qy,qz,qw)
    anything else      → Cartesian pose  → PlanToPose

  Planner profile (determined by value prefix token):
    _Lin  → linear_approach   (Pilz LIN – straight line EE motion)
    _PTP  → fast_ptp          (Pilz PTP – fast point-to-point, default)
    (none) → fast_ptp         (default)

  Speed section:
    speed:<section_name>: vel, acc, [lin_vel, lin_acc]
    → overrides velocity_scaling on each node in that section.

Usage:
    python3 sequence_to_bt.py \
        --yaml  <path/to/sequence.yaml> \
        --out   <path/to/output.xml> \
        [--section homePoses]          # optional: only convert one section
        [--arm left_arm]               # override active arm for all nodes

    Example:
        python3 sequence_to_bt.py \
            --yaml config/sequence.yaml \
            --out  config/demo_sequence.xml
"""

import argparse
import re
import sys
import xml.dom.minidom as minidom
from pathlib import Path
from typing import Optional

import yaml  # pip install pyyaml


# ──────────────────────────────────────────────────────────────────────────────
# Key-rule helpers
# ──────────────────────────────────────────────────────────────────────────────

ARM_PREFIX = {
    "la": "left_arm",
    "ra": "right_arm",
    "lh": "left_gripper",
    "rh": "right_gripper",
}

PLANNER_TOKEN = {
    "_Lin": "linear_approach",
    "_PTP": "fast_ptp",
}

# Joint-name suffix used in key naming (contains Angle → joint, contains Joint → pose-labeled-as-joint)
_KEY_IS_JOINT_ANGLE = re.compile(r"Angle", re.IGNORECASE)
_KEY_IS_POSE        = re.compile(r"Joint",  re.IGNORECASE)


def arm_from_key(key: str) -> Optional[str]:
    """Return the MoveIt planning group for a key, or None if not an arm key."""
    for prefix, group in ARM_PREFIX.items():
        if key.lower().startswith(prefix.lower()):
            return group
    return None


def planner_from_values(tokens: list[str]) -> tuple[str, list[str]]:
    """
    Strip planner tokens from the front of the value list.
    Returns (profile_name, remaining_tokens).
    """
    profile = "fast_ptp"
    remaining = list(tokens)
    for token_key, prof in PLANNER_TOKEN.items():
        if remaining and remaining[0].strip() == token_key:
            profile = prof
            remaining = remaining[1:]
            break
    return profile, remaining


def is_joint_angle(key: str) -> bool:
    return bool(_KEY_IS_JOINT_ANGLE.search(key))


def parse_value(raw: str) -> list[str]:
    """Split a comma-separated value string into tokens."""
    return [t.strip() for t in raw.split(",") if t.strip() != ""]


def speed_for_section(speed_dict: dict, section: str) -> Optional[float]:
    """Return the velocity scale for a section, or None if not defined."""
    if not speed_dict:
        return None
    val = speed_dict.get(section)
    if val is None:
        return None
    tokens = parse_value(str(val))
    try:
        return float(tokens[0])
    except (IndexError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# XML builders
# ──────────────────────────────────────────────────────────────────────────────

def make_set_planner(doc: minidom.Document, profile: str) -> minidom.Element:
    el = doc.createElement("SetPlannerConfig")
    el.setAttribute("profile", profile)
    return el


def make_plan_to_named_pose(doc: minidom.Document, arm: str, pose_name: str,
                             vel: Optional[float] = None) -> minidom.Element:
    """
    PlanToNamedPose: arm moves to a named pose from the SRDF / blackboard.
    Used for joint-angle keys (the key name becomes the SRDF state name).
    """
    el = doc.createElement("PlanToNamedPose")
    el.setAttribute("arm", arm)
    el.setAttribute("pose_name", pose_name)
    el.setAttribute("duration", "3.0")
    el.setAttribute("output_trajectory", "{plan_trajectory}")
    if vel is not None:
        el.setAttribute("velocity_scaling", f"{vel:.2f}")
    return el


def make_plan_to_pose(doc: minidom.Document, arm: str, pose_name: str,
                       values: list[str], vel: Optional[float] = None,
                       position_only: bool = False) -> minidom.Element:
    """
    PlanToPose: arm moves to a Cartesian pose [x, y, z, qx, qy, qz, qw].
    The pose is embedded as a comment + uses BB key injected at runtime.
    For static poses from the YAML, we write them into a blackboard-init
    comment block so the user knows what to wire.
    """
    el = doc.createElement("PlanToPose")
    el.setAttribute("arm", arm)
    el.setAttribute("target_pose", f"{{{pose_name}_pose}}")  # BB key by convention
    el.setAttribute("velocity_scaling", f"{vel:.2f}" if vel is not None else "0.3")
    el.setAttribute("position_only", "true" if position_only else "false")
    el.setAttribute("output_trajectory", "{plan_trajectory}")
    # Embed raw values as XML attribute comment so user can see the numbers
    pose_str = ", ".join(values)
    comment = doc.createComment(
        f' {pose_name}: [{pose_str}]  → set BB key "{pose_name}_pose" before this node '
    )
    return el, comment


def make_close_gripper(doc: minidom.Document, arm: str) -> minidom.Element:
    el = doc.createElement("CloseGripper")
    # arm is "left_gripper" / "right_gripper" — map back to arm name for port
    arm_name = "left_arm" if "left" in arm else "right_arm"
    el.setAttribute("arm", arm_name)
    el.setAttribute("close_position", "0.0")
    el.setAttribute("duration", "1.5")
    return el


def make_open_gripper(doc: minidom.Document, arm: str) -> minidom.Element:
    el = doc.createElement("OpenGripper")
    arm_name = "left_arm" if "left" in arm else "right_arm"
    el.setAttribute("arm", arm_name)
    el.setAttribute("open_position", "0.044")
    el.setAttribute("duration", "1.0")
    return el


# ──────────────────────────────────────────────────────────────────────────────
# Section converter
# ──────────────────────────────────────────────────────────────────────────────

def convert_section(doc: minidom.Document, section_name: str, section_data: dict,
                    speed_dict: dict, arm_override: Optional[str]) -> minidom.Element:
    """Convert one YAML section into a <BehaviorTree> element."""

    bt_el = doc.createElement("BehaviorTree")
    bt_id = to_pascal_case(section_name)
    bt_el.setAttribute("ID", bt_id)

    seq = doc.createElement("Sequence")
    seq.setAttribute("name", section_name)
    bt_el.appendChild(seq)

    # Default velocity for this section
    section_vel = speed_for_section(speed_dict, section_name)

    last_profile = None  # avoid duplicate SetPlannerConfig nodes

    for key, raw_val in section_data.items():
        if raw_val is None:
            continue

        tokens = parse_value(str(raw_val))
        if not tokens:
            continue

        arm = arm_override or arm_from_key(key)

        # ── Gripper keys ───────────────────────────────────────────────────
        if arm in ("left_gripper", "right_gripper"):
            # Detect open/close by key name
            if "open" in key.lower() or "release" in key.lower():
                seq.appendChild(doc.createComment(f" {key} "))
                seq.appendChild(make_open_gripper(doc, arm))
            else:
                seq.appendChild(doc.createComment(f" {key} "))
                seq.appendChild(make_close_gripper(doc, arm))
            continue

        # ── Head keys — skip (not a MoveIt planning group here) ───────────
        if key.lower().startswith("head"):
            seq.appendChild(doc.createComment(
                f" {key}: head move (not yet wired to BT node) — values: {', '.join(tokens)} "))
            continue

        # ── Arm keys ──────────────────────────────────────────────────────
        if arm is None:
            # Generic / scalar config value — skip silently or comment
            seq.appendChild(doc.createComment(
                f" config: {key} = {', '.join(tokens)} "))
            continue

        profile, value_tokens = planner_from_values(tokens)
        if not value_tokens:
            continue

        # Emit SetPlannerConfig only when profile changes
        if profile != last_profile:
            seq.appendChild(make_set_planner(doc, profile))
            last_profile = profile

        if is_joint_angle(key):
            # Joint-angle key → PlanToNamedPose using key name as the SRDF state name
            seq.appendChild(doc.createComment(
                f" {key}: joint angles [{', '.join(value_tokens)}] "
                f"→ add this as SRDF named_state or extend SkillRequest with joint_targets "))
            node = make_plan_to_named_pose(doc, arm, key, section_vel)
            seq.appendChild(node)
        else:
            # Cartesian pose key → PlanToPose
            node, comment = make_plan_to_pose(doc, arm, key, value_tokens, section_vel)
            seq.appendChild(comment)
            seq.appendChild(node)

    return bt_el


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def to_pascal_case(s: str) -> str:
    return "".join(word.capitalize() for word in re.split(r"[_\s]+", s))


def pretty_xml(doc: minidom.Document) -> str:
    raw = doc.toprettyxml(indent="  ", encoding=None)
    # Remove the extra blank lines minidom adds
    lines = [l for l in raw.splitlines() if l.strip()]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main converter
# ──────────────────────────────────────────────────────────────────────────────

def convert(yaml_path: Path, out_path: Path,
            section_filter: Optional[str] = None,
            arm_override: Optional[str] = None) -> None:

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        print(f"[ERROR] YAML file is empty: {yaml_path}")
        sys.exit(1)

    speed_dict = data.pop("speed", {}) or {}

    # Sections to convert
    sections = {}
    if section_filter:
        if section_filter not in data:
            print(f"[ERROR] Section '{section_filter}' not found in YAML.")
            print(f"Available sections: {list(data.keys())}")
            sys.exit(1)
        sections[section_filter] = data[section_filter]
    else:
        sections = {k: v for k, v in data.items() if isinstance(v, dict)}

    if not sections:
        print("[ERROR] No convertible sections (dict values) found in YAML.")
        sys.exit(1)

    # Build XML document
    doc = minidom.Document()
    root = doc.createElement("root")
    root.setAttribute("BTCPP_format", "4")

    # main_tree = first section (or the filtered one)
    first_section = to_pascal_case(next(iter(sections)))
    root.setAttribute("main_tree_to_execute", first_section)
    doc.appendChild(root)

    doc.insertBefore(
        doc.createComment(
            f" Auto-generated from {yaml_path.name} by sequence_to_bt.py.\n"
            "     DO NOT edit manually — regenerate from the YAML instead.\n"
            "     Wire BB keys for Cartesian poses before executing. "
        ),
        root,
    )

    for section_name, section_data in sections.items():
        if not section_data:
            continue
        bt_el = convert_section(doc, section_name, section_data, speed_dict, arm_override)
        root.appendChild(bt_el)

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xml_str = pretty_xml(doc)
    with open(out_path, "w") as f:
        f.write(xml_str + "\n")

    print(f"[OK] Written {out_path}")
    print(f"     Sections converted: {list(sections.keys())}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenArm sequence.yaml → BehaviorTree.CPP v4 XML"
    )
    parser.add_argument(
        "--yaml", required=True, type=Path,
        help="Path to sequence.yaml"
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output XML file path"
    )
    parser.add_argument(
        "--section", default=None,
        help="Convert only one named section (e.g. graspPoses)"
    )
    parser.add_argument(
        "--arm", default=None,
        help="Override planning group for all nodes (e.g. left_arm)"
    )
    args = parser.parse_args()

    if not args.yaml.exists():
        print(f"[ERROR] YAML file not found: {args.yaml}")
        sys.exit(1)

    convert(args.yaml, args.out, args.section, args.arm)


if __name__ == "__main__":
    main()
