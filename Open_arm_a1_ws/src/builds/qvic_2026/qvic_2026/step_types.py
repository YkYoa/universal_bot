"""The step-type catalog: the single source of truth for what a sequence step
can be.

Three consumers read this one table:
  - store.py, to validate params before a row is written
  - the Flask blueprint's GET /api/step-types, which is what the Android app
    builds its drag-and-drop palette from (so the palette is never hardcoded
    on the client)
  - the C++ SequenceFsm, indirectly - it dispatches on `type` and reads the
    same param names out of params_json

Adding a step type means adding an entry here, teaching store.py nothing new,
and adding one dispatch branch in sequence_fsm.cpp.

`control_mode` is the hardware mode a step needs. The arm's control mode is
fixed at startup (the damiao register is written once during init, see
robot_hardware_interface/src/v10/hardware_interface.cpp), so this is validated,
never switched: a "position|mit" step is rejected when the arm came up in
torque, and a "torque" step is rejected when it came up in position.
"""

# Control-mode requirement values.
MODE_ANY = "any"
MODE_MOTION = "position|mit"   # anything that commands a joint trajectory
MODE_TORQUE = "torque"         # gravity-comp / hand-guiding only

ARMS = ("left_arm", "right_arm", "both_arms")
SIDES = ("left", "right")


def _f(name, type_, required=True, default=None, desc="", **extra):
    field = {
        "name": name,
        "type": type_,
        "required": required,
        "default": default,
        "description": desc,
    }
    field.update(extra)
    return field


# Every motion step can override the sequence's scaling. 0 means "inherit":
# first the sequence's own value, then the planner profile's default. This is
# where sequence.yaml's per-section `speed:` map lands on import.
_SPEED_FIELDS = (
    _f("velocity", "float", required=False, default=0.0,
       desc="MoveIt velocity scaling 0-1. 0 = inherit.", minimum=0.0, maximum=1.0),
    _f("acceleration", "float", required=False, default=0.0,
       desc="MoveIt acceleration scaling 0-1. 0 = inherit.", minimum=0.0, maximum=1.0),
)

# type -> {label, description, control_mode, fields[]}
STEP_TYPES = {
    "move_joint": {
        "label": "Move to waypoint",
        "description": "Plan and move one arm to a single recorded joint waypoint.",
        "control_mode": MODE_MOTION,
        "fields": [
            _f("arm", "enum", desc="Which arm moves.", options=list(ARMS)),
            _f("waypoint", "waypoint_ref", required=False,
               desc="Stored *Angle waypoint as 'section/name' (names repeat across "
                    "sections, so the section qualifier matters). Mutually exclusive "
                    "with `positions`."),
            _f("right_waypoint", "waypoint_ref", required=False,
               desc="For arm=both_arms: the right-arm 'section/name', appended after `waypoint`."),
            _f("positions", "float[]", required=False,
               desc="Raw joint values instead of a waypoint name. 7 for one arm, 14 for both_arms.",
               length=[7, 14]),
            *_SPEED_FIELDS,
        ],
    },
    "move_joint_sequence": {
        "label": "Play waypoint section",
        "description": (
            "Replay a whole section of waypoints as one blended joint sequence "
            "(the corner-blended PlanToJointSequence path)."
        ),
        "control_mode": MODE_MOTION,
        "fields": [
            _f("arm", "enum", desc="Which arm moves.", options=list(ARMS)),
            _f("section", "section_ref", desc="Waypoint section to replay."),
            _f("right_section", "section_ref", required=False,
               desc="For both_arms: the right-arm section, interleaved 1:1 with `section`. "
                    "Waypoint counts must match."),
            _f("exclude_points", "int[]", required=False, default=[],
               desc="1-indexed waypoints to skip - the escape hatch for a known-bad point."),
            *_SPEED_FIELDS,
        ],
    },
    "move_pose": {
        "label": "Move to Cartesian pose",
        "description": "Plan and move one arm to an (x, y, z) + quaternion pose.",
        "control_mode": MODE_MOTION,
        "fields": [
            _f("arm", "enum", desc="Which arm moves.", options=list(ARMS)),
            _f("position", "float[]", desc="x, y, z in metres.", length=[3]),
            _f("orientation", "float[]", required=False,
               desc="qx, qy, qz, qw. Omit together with position_only=true.", length=[4]),
            _f("position_only", "bool", required=False, default=False,
               desc="Ignore orientation when planning."),
            _f("frame_id", "string", required=False, default="openarm_body_link0",
               desc="Frame the pose is expressed in."),
            *_SPEED_FIELDS,
        ],
    },
    "named_pose": {
        "label": "Move to named pose",
        "description": "Move a planning group to a pose named in the SRDF.",
        "control_mode": MODE_MOTION,
        "fields": [
            _f("group", "string", desc="MoveIt planning group."),
            _f("pose", "string", desc="SRDF named pose."),
            *_SPEED_FIELDS,
        ],
    },
    "hand_pose": {
        "label": "Hand pose",
        "description": (
            "Set amazing_hand yaw and/or flex. Every vector supplied fires "
            "concurrently - left and right, yaw and flex - so one step is one "
            "simultaneous hand posture. Supply only the vectors you want moved."
        ),
        "control_mode": MODE_ANY,
        "fields": [
            _f("left_yaw", "float[]", required=False,
               desc="4 left yaw values (finger 1-4, 4 = thumb).", length=[4]),
            _f("left_flex", "float[]", required=False, desc="4 left flex values.", length=[4]),
            _f("right_yaw", "float[]", required=False, desc="4 right yaw values.", length=[4]),
            _f("right_flex", "float[]", required=False, desc="4 right flex values.", length=[4]),
            _f("duration", "float", required=False, default=1.0, desc="Move duration in seconds."),
            _f("section", "string", required=False,
               desc="Waypoint section these values came from. Informational - lets "
                    "YAML export rebuild a body_sections list."),
        ],
    },
    "gripper": {
        "label": "Gripper",
        "description": "Open or close the parallel gripper (openarm_hand end effector).",
        "control_mode": MODE_ANY,
        "fields": [
            _f("side", "enum", desc="Which gripper.", options=list(SIDES)),
            _f("action", "enum", desc="open or close.", options=["open", "close"]),
        ],
    },
    "wait": {
        "label": "Wait",
        "description": "Hold position for a fixed time.",
        "control_mode": MODE_ANY,
        "fields": [
            _f("seconds", "float", desc="How long to wait.", minimum=0.0),
        ],
    },
    "attach_object": {
        "label": "Attach object (grasp)",
        "description": (
            "Attach a collision object to a link. MoveIt then carries it with "
            "the arm and stops flagging it against `touch_links` - this is the "
            "'allow collision while grasping' step."
        ),
        "control_mode": MODE_ANY,
        "fields": [
            _f("object_id", "string", desc="Id of an object already in the scene."),
            _f("link", "string", desc="Link to attach it to, e.g. openarm_left_hand_tcp."),
            _f("touch_links", "string[]", required=False, default=[],
               desc="Links allowed to touch it. Empty = every link of that hand group."),
        ],
    },
    "detach_object": {
        "label": "Detach object (release)",
        "description": "Detach an object, leaving it in the scene where it currently is.",
        "control_mode": MODE_ANY,
        "fields": [
            _f("object_id", "string", desc="Id of the attached object."),
            _f("link", "string", required=False, desc="Link it is attached to."),
        ],
    },
    "add_object": {
        "label": "Add collision object",
        "description": "Put a primitive obstacle into the planning scene.",
        "control_mode": MODE_ANY,
        "fields": [
            _f("object_id", "string", desc="Id to give it."),
            _f("primitive", "enum", desc="Shape.", options=["box", "sphere", "cylinder"]),
            _f("dimensions", "float[]", desc="box [x,y,z], sphere [r], cylinder [h,r].", length=[1, 2, 3]),
            _f("position", "float[]", desc="x, y, z in metres.", length=[3]),
            _f("orientation", "float[]", required=False, default=[0.0, 0.0, 0.0, 1.0],
               desc="qx, qy, qz, qw.", length=[4]),
            _f("frame_id", "string", required=False, default="openarm_body_link0",
               desc="Frame the pose is expressed in."),
        ],
    },
    "remove_object": {
        "label": "Remove collision object",
        "description": "Delete an object from the planning scene.",
        "control_mode": MODE_ANY,
        "fields": [
            _f("object_id", "string", desc="Id of the object to remove."),
        ],
    },
    "allow_collision": {
        "label": "Allow collision",
        "description": (
            "Directly set AllowedCollisionMatrix entries. Use for the surface an "
            "object rests on; prefer attach_object for the grasp itself."
        ),
        "control_mode": MODE_ANY,
        "fields": [
            _f("object_id", "string", desc="One side of the pair."),
            _f("links", "string[]", required=False, default=[],
               desc="Other side. Empty = allow against everything."),
        ],
    },
    "disallow_collision": {
        "label": "Disallow collision",
        "description": "Undo allow_collision.",
        "control_mode": MODE_ANY,
        "fields": [
            _f("object_id", "string", desc="One side of the pair."),
            _f("links", "string[]", required=False, default=[], desc="Other side."),
        ],
    },
    "set_speed": {
        "label": "Set speed",
        "description": "Change velocity/acceleration scaling for every following step.",
        "control_mode": MODE_ANY,
        "fields": [
            _f("velocity", "float", desc="MoveIt velocity scaling 0-1. 0 = profile default.",
               minimum=0.0, maximum=1.0),
            _f("acceleration", "float", required=False, default=0.0,
               desc="MoveIt acceleration scaling 0-1. 0 = profile default.",
               minimum=0.0, maximum=1.0),
        ],
    },
    "teach_hold": {
        "label": "Hand-guide hold",
        "description": (
            "Hold in gravity compensation so a human can move the arm by hand. "
            "Only valid when the arm came up in torque mode."
        ),
        "control_mode": MODE_TORQUE,
        "fields": [
            _f("seconds", "float", desc="How long to stay compliant.", minimum=0.0),
        ],
    },
}


def catalog():
    """The palette, as the API hands it to Android."""
    return {
        name: {
            "label": spec["label"],
            "description": spec["description"],
            "control_mode": spec["control_mode"],
            "fields": spec["fields"],
        }
        for name, spec in STEP_TYPES.items()
    }


def control_mode_for(step_type):
    spec = STEP_TYPES.get(step_type)
    return spec["control_mode"] if spec else MODE_ANY


def mode_is_compatible(required, active):
    """Does a step needing `required` run on hardware in `active` mode?

    `required` is one of the MODE_* constants; `active` is what the hardware
    actually came up in. Unknown active mode is permissive - the probe failing
    should not brick every sequence.
    """
    if required == MODE_ANY or not active or active == "unknown":
        return True
    return active in required.split("|")


class StepValidationError(ValueError):
    """Raised when a step's params don't match its type's schema."""


def validate_step(step_type, params):
    """Check `params` against the type's field list. Returns params with
    defaults filled in; raises StepValidationError on anything wrong."""
    spec = STEP_TYPES.get(step_type)
    if spec is None:
        raise StepValidationError(
            f"unknown step type '{step_type}'; known types: {sorted(STEP_TYPES)}"
        )
    if not isinstance(params, dict):
        raise StepValidationError("params must be an object")

    known = {f["name"] for f in spec["fields"]}
    unknown = set(params) - known
    if unknown:
        raise StepValidationError(
            f"{step_type}: unknown param(s) {sorted(unknown)}; allowed: {sorted(known)}"
        )

    out = {}
    for field in spec["fields"]:
        name = field["name"]
        if name not in params or params[name] is None:
            if field["required"]:
                raise StepValidationError(f"{step_type}: missing required param '{name}'")
            if field["default"] is not None:
                out[name] = field["default"]
            continue
        out[name] = _coerce(step_type, field, params[name])

    _check_pairs(step_type, out)
    return out


def _check_pairs(step_type, params):
    """The either/or rules that a per-field `required` flag can't express."""
    if step_type == "move_joint":
        has_wp = bool(params.get("waypoint"))
        has_pos = bool(params.get("positions"))
        if has_wp == has_pos:
            raise StepValidationError(
                "move_joint: set exactly one of 'waypoint' or 'positions'"
            )
        if params.get("right_waypoint") and params["arm"] != "both_arms":
            raise StepValidationError(
                "move_joint: 'right_waypoint' only applies when arm is both_arms"
            )
        if params["arm"] == "both_arms" and has_pos and len(params["positions"]) != 14:
            raise StepValidationError(
                "move_joint: both_arms needs 14 positions (left 7 then right 7)"
            )
        if params["arm"] != "both_arms" and has_pos and len(params["positions"]) != 7:
            raise StepValidationError(f"move_joint: {params['arm']} needs 7 positions")
    if step_type == "move_pose":
        if not params.get("position_only") and "orientation" not in params:
            raise StepValidationError(
                "move_pose: needs 'orientation', or 'position_only': true"
            )
    if step_type == "hand_pose":
        if not any(k in params for k in ("left_yaw", "left_flex", "right_yaw", "right_flex")):
            raise StepValidationError(
                "hand_pose: needs at least one of left_yaw, left_flex, right_yaw, right_flex"
            )
    if step_type == "move_joint_sequence":
        if params.get("right_section") and params["arm"] != "both_arms":
            raise StepValidationError(
                "move_joint_sequence: 'right_section' only applies when arm is both_arms"
            )


def _coerce(step_type, field, value):
    name, kind = field["name"], field["type"]
    where = f"{step_type}.{name}"

    if kind == "enum":
        if value not in field["options"]:
            raise StepValidationError(f"{where}: '{value}' not in {field['options']}")
        return value

    if kind == "bool":
        if not isinstance(value, bool):
            raise StepValidationError(f"{where}: expected true/false")
        return value

    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StepValidationError(f"{where}: expected a number")
        value = float(value)
        if "minimum" in field and value < field["minimum"]:
            raise StepValidationError(f"{where}: must be >= {field['minimum']}")
        if "maximum" in field and value > field["maximum"]:
            raise StepValidationError(f"{where}: must be <= {field['maximum']}")
        return value

    if kind in ("string", "waypoint_ref", "section_ref"):
        if not isinstance(value, str) or not value.strip():
            raise StepValidationError(f"{where}: expected a non-empty string")
        return value.strip()

    if kind == "float[]":
        if not isinstance(value, (list, tuple)) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in value
        ):
            raise StepValidationError(f"{where}: expected a list of numbers")
        if "length" in field and len(value) not in field["length"]:
            raise StepValidationError(
                f"{where}: expected {' or '.join(str(n) for n in field['length'])} "
                f"values, got {len(value)}"
            )
        return [float(v) for v in value]

    if kind == "int[]":
        if not isinstance(value, (list, tuple)) or any(
            isinstance(v, bool) or not isinstance(v, int) for v in value
        ):
            raise StepValidationError(f"{where}: expected a list of integers")
        return list(value)

    if kind == "string[]":
        if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) for v in value):
            raise StepValidationError(f"{where}: expected a list of strings")
        return list(value)

    raise StepValidationError(f"{where}: unhandled field type '{kind}'")
