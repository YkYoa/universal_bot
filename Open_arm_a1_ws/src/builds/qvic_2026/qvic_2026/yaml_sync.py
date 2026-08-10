"""Two-way bridge between config/sequence.yaml and the SQLite store.

The YAML is not dead: waypoint_recorder and trajectory_waypoint_generator both
write it, and it is what git diffs. So the workflow stays

    record/generate -> sequence.yaml -> import -> store -> Android edits it
                                      <- export <-

The parsing rules here mirror control/sequence_executor/src/sequence_yaml.cpp
exactly, because the same files have to mean the same thing to both:

  key contains "Angle"  -> 7 joint values, a replayable waypoint
  key matches ^(lh|rh).*Yaw$ / Flex$ -> 4 hand values
  any other key         -> a Cartesian reference pose (3 position + 4 quat)
  top-level `speed:`    -> per-section velocity[, acceleration]
  top-level `sequences:` -> the flat sequence definitions
"""

import re

import yaml

from . import store

ANGLE_RE = re.compile(r"angle", re.IGNORECASE)
HAND_YAW_RE = re.compile(r"^(lh|rh).*Yaw$")
HAND_FLEX_RE = re.compile(r"^(lh|rh).*Flex$")
ARM_PREFIX_RE = re.compile(r"^(la|ra|lh|rh|head)")

# Keys that are sequence bookkeeping, not waypoint sections.
RESERVED_SECTIONS = ("speed", "sequences")

ARM_TO_PREFIX = {"left_arm": "la", "right_arm": "ra"}


def _values(raw):
    """sequence.yaml writes vectors as a comma-separated scalar, but a real
    YAML list is also accepted (both forms appear in the tree)."""
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw]
    return [float(tok) for tok in str(raw).split(",") if tok.strip()]


def _kind_for(key):
    if HAND_YAW_RE.match(key):
        return "hand_yaw"
    if HAND_FLEX_RE.match(key):
        return "hand_flex"
    if ANGLE_RE.search(key):
        return "angle"
    return "pose"


def _prefix_for(key):
    m = ARM_PREFIX_RE.match(key)
    return m.group(1) if m else ""


# ── import ────────────────────────────────────────────────────────────────

def import_yaml(yaml_path, path=None, replace=True):
    """Read a sequence.yaml into the store.

    Returns a summary dict. `replace=True` overwrites sequences that already
    exist by name; False skips them.
    """
    with open(yaml_path, "r") as f:
        root = yaml.safe_load(f) or {}
    if not isinstance(root, dict):
        raise store.StoreError(f"{yaml_path} is not a YAML mapping")

    speeds = _parse_speeds(root.get("speed") or {})
    waypoint_count = 0
    sections = {}
    bad = []

    for section, body in root.items():
        if section in RESERVED_SECTIONS or not isinstance(body, dict):
            continue
        sections[section] = body
        for key, raw in body.items():
            try:
                values = _values(raw)
            except ValueError as exc:
                # Never skip quietly. The C++ reader parses with std::stod,
                # which stops at the first bad character and hands the planner
                # a truncated-but-plausible joint value; a corrupted line has
                # to fail loudly here instead.
                bad.append(f"{section}/{key}: {exc}")
                continue
            if not values:
                bad.append(f"{section}/{key}: empty value")
                continue
            store.upsert_waypoint(
                name=key, section=section, arm_prefix=_prefix_for(key),
                kind=_kind_for(key), values=values, path=path,
            )
            waypoint_count += 1

    if bad:
        raise store.StoreError(
            f"{yaml_path} has {len(bad)} unparseable waypoint value(s):\n  "
            + "\n  ".join(bad)
        )

    imported, skipped = [], []
    for name, node in (root.get("sequences") or {}).items():
        if not isinstance(node, dict):
            continue
        try:
            steps = _steps_for(node, sections, speeds)
        except store.StoreError as exc:
            raise store.StoreError(f"sequences:{name}: {exc}") from exc

        existing = None
        try:
            existing = store.get_sequence(name, path=path)
        except store.StoreError:
            pass
        if existing and not replace:
            skipped.append(name)
            continue

        fields = dict(
            description=f"imported from {yaml_path}",
            arm=node.get("arm", "left_arm"),
            planner_profile=node.get("planner_profile", ""),
            repeat=int(node.get("repeat", 1)),
        )
        if existing:
            store.update_sequence(name, path=path, **fields)
            store.replace_steps(name, steps, path=path)
        else:
            store.create_sequence(name, steps=steps, path=path, **fields)
        imported.append(name)

    return {
        "waypoints": waypoint_count,
        "sections": len(sections),
        "sequences": imported,
        "skipped": skipped,
    }


def _parse_speeds(node):
    out = {}
    for section, raw in (node or {}).items():
        try:
            vals = _values(raw)
        except ValueError:
            continue
        out[section] = (
            vals[0] if vals else 0.0,
            vals[1] if len(vals) > 1 else 0.0,
        )
    return out


def _steps_for(node, sections, speeds):
    """Flatten one `sequences:` entry into an ordered step list.

    The YAML schema is fixed-shape - home once, then a body that loops - so
    this is where that shape is unrolled into the generic step list the FSM
    now walks.
    """
    arm = node.get("arm", "left_arm")
    steps = []

    home = node.get("home_section")
    if home:
        if home not in sections:
            raise store.StoreError(f"home_section '{home}' has no matching section")
        steps.extend(_home_steps(home, sections[home], arm, speeds))

    body_sections = _tokens(node.get("body_sections"))
    body = node.get("body_section")

    if body_sections:
        for section in body_sections:
            if section not in sections:
                raise store.StoreError(f"body_sections entry '{section}' has no matching section")
            hand = _hand_step(section, sections[section])
            if hand:
                steps.append(hand)
    elif body:
        if body not in sections:
            raise store.StoreError(f"body_section '{body}' has no matching section")
        vel, acc = speeds.get(body, (0.0, 0.0))
        params = {"arm": arm, "section": body, "velocity": vel, "acceleration": acc}
        right = node.get("body_right_section")
        if right:
            if right not in sections:
                raise store.StoreError(f"body_right_section '{right}' has no matching section")
            params["right_section"] = right
        excluded = [int(t) for t in _tokens(node.get("exclude_points"))]
        if excluded:
            params["exclude_points"] = excluded
        steps.append({"name": f"Play {body}", "type": "move_joint_sequence", "params": params})
    else:
        raise store.StoreError("needs either 'body_section' or 'body_sections'")

    return steps


def _home_steps(section_name, body, arm, speeds):
    """The YAML home step is an arm move plus, if the section carries hand
    keys, a hand hold - matching runHome()'s arm-then-hand order."""
    steps = []
    vel, acc = speeds.get(section_name, (0.0, 0.0))

    if arm == "both_arms":
        left = _find(section_name, body, "laHomeAngle")
        right = _find(section_name, body, "raHomeAngle")
        if left and right:
            steps.append({
                "name": f"Home ({section_name})",
                "type": "move_joint",
                "params": {"arm": arm, "waypoint": left, "right_waypoint": right,
                           "velocity": vel, "acceleration": acc},
            })
    else:
        key = _find(section_name, body, ARM_TO_PREFIX.get(arm, "la") + "HomeAngle")
        if key:
            steps.append({
                "name": f"Home ({section_name})",
                "type": "move_joint",
                "params": {"arm": arm, "waypoint": key,
                           "velocity": vel, "acceleration": acc},
            })

    hand = _hand_step(section_name, body)
    if hand:
        steps.append(hand)
    return steps


def _hand_step(section_name, body):
    """Collapse a section's lh/rh Yaw+Flex keys into one concurrent hand_pose
    step - the same fan-out runHandPoseSection() does."""
    params = {}
    for key, raw in body.items():
        if HAND_YAW_RE.match(key):
            params["left_yaw" if key.startswith("lh") else "right_yaw"] = _values(raw)
        elif HAND_FLEX_RE.match(key):
            params["left_flex" if key.startswith("lh") else "right_flex"] = _values(raw)
    if not params:
        return None
    params["duration"] = 1.0     # kHandMoveDurationS in sequence_interpreter.cpp
    params["section"] = section_name
    return {"name": f"Hand ({section_name})", "type": "hand_pose", "params": params}


def _find(section, body, key):
    """Waypoint refs are section-qualified because sequence.yaml reuses names
    (laHomeAngle lives in both homePoses and waveHome)."""
    return store.waypoint_ref(section, key) if key in body else None


def _tokens(raw):
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(v).strip() for v in raw if str(v).strip()]
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


# ── export ────────────────────────────────────────────────────────────────

def export_yaml(yaml_path, path=None):
    """Write the store back out in sequence.yaml form.

    Round-trips the data, not the comments - the source file's header block is
    documentation for humans, so exporting is for feeding the CLI tools and for
    diffing, not for replacing the checked-in file wholesale.
    """
    waypoints = store.list_waypoints(path=path)
    sequences = store.list_sequences(path=path)

    root = {}
    speed = {}
    for wp in waypoints:
        root.setdefault(wp["section"], {})[wp["name"]] = ", ".join(
            _fmt(v) for v in wp["values"]
        )

    seq_block = {}
    for summary in sequences:
        if summary["builtin"]:
            continue
        seq = store.get_sequence(summary["name"], path=path)
        entry, seq_speed = _sequence_to_yaml(seq)
        if entry:
            seq_block[seq["name"]] = entry
            speed.update(seq_speed)

    out = {}
    if speed:
        out["speed"] = speed
    out.update(root)
    if seq_block:
        out["sequences"] = seq_block

    with open(yaml_path, "w") as f:
        f.write(
            "# Generated by `sequence_store export`. The store is the source of\n"
            "# truth; this file exists so waypoint_recorder,\n"
            "# trajectory_waypoint_generator, and git diffs keep working.\n\n"
        )
        yaml.safe_dump(out, f, default_flow_style=False, sort_keys=False)
    return {"waypoints": len(waypoints), "sequences": list(seq_block)}


def _sequence_to_yaml(seq):
    """Best-effort inverse of _steps_for. A sequence built in the Android app
    can hold steps the flat YAML schema cannot express (wait, attach_object,
    per-step speed changes); those are dropped, and the entry is skipped
    entirely if nothing replayable survives."""
    entry = {"arm": seq["arm"]}
    if seq["planner_profile"]:
        entry["planner_profile"] = seq["planner_profile"]

    steps = [s for s in seq["steps"] if s["enabled"]]
    speed = {}
    body_hand_sections = []
    home_hand_index = None

    for i, step in enumerate(steps):
        params = step["params"]

        if step["type"] == "move_joint" and "home_section" not in entry:
            section, _ = store.split_ref(params.get("waypoint") or "")
            if section:
                entry["home_section"] = section
                # The YAML home step is arm-then-hand, so a hand_pose sitting
                # immediately after it from the same section belongs to home -
                # counting it as a body section would replay it twice.
                nxt = steps[i + 1] if i + 1 < len(steps) else None
                if nxt and nxt["type"] == "hand_pose" and \
                        nxt["params"].get("section") == section:
                    home_hand_index = i + 1

        elif step["type"] == "move_joint_sequence":
            entry["body_section"] = params["section"]
            if params.get("right_section"):
                entry["body_right_section"] = params["right_section"]
            if params.get("exclude_points"):
                entry["exclude_points"] = ", ".join(str(i) for i in params["exclude_points"])
            if params.get("velocity"):
                speed[params["section"]] = _fmt(params["velocity"]) + (
                    ", " + _fmt(params["acceleration"]) if params.get("acceleration") else ""
                )

        elif step["type"] == "hand_pose" and i != home_hand_index:
            section = params.get("section")
            if section:
                body_hand_sections.append(section)

    if "body_section" not in entry and body_hand_sections:
        # A hand-only animation maps onto body_sections - exactly what the
        # hand_open_close entry in sequence.yaml does.
        entry["body_sections"] = ", ".join(body_hand_sections)

    if "body_section" not in entry and "body_sections" not in entry:
        return None, {}
    if seq["repeat"] != 1:
        entry["repeat"] = seq["repeat"]
    return entry, speed


def _fmt(value):
    """Match the source YAML's plain decimal style - no scientific notation,
    no trailing zero noise."""
    text = f"{float(value):.12g}"
    return text
