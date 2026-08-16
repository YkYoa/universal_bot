"""Shared, ee_type-aware loader for openarm_bimanual.srdf.

The static SRDF file contains groups/group_states/disable_collisions for
BOTH end-effector types (openarm_hand's left_gripper/right_gripper,
amazing_hand's left_hand_fingers/right_hand_fingers) - only one set is ever
valid for a given ee_type, since the other set's joints don't exist in that
URDF. Loading the raw file unconditionally causes MoveIt to log
"Group '...' not found in model" and can crash RViz's MotionPlanningDisplay
trying to build markers for the dangling end-effector group.

load_srdf_for_ee_type() strips whichever set doesn't apply, so every launch
file that builds robot_description_semantic from this SRDF gets a
consistent, correct-for-both-directions patch instead of each hand-rolling
its own (previously one-directional, previously copy-pasted three times).
"""

import re

import yaml


def load_srdf_for_ee_type(srdf_path: str, ee_type: str, body_type: str = None) -> str:
    with open(srdf_path, "r") as f:
        content = f.read()

    if ee_type == "amazing_hand":
        content = (
            content
            .replace('group="left_gripper" parent_group="left_arm"',
                      'group="left_hand_fingers" parent_group="left_arm"')
            .replace('group="right_gripper" parent_group="right_arm"',
                      'group="right_hand_fingers" parent_group="right_arm"')
        )
        # The rest of the SRDF (left_gripper/right_gripper groups, their
        # open/close group_states, and disable_collisions entries for
        # openarm_*_hand/*_finger) is written for ee_type:=openarm_hand's
        # 2-finger gripper links, which don't exist under amazing_hand.
        content = re.sub(
            r'  <group name="(?:left|right)_gripper">.*?</group>\n\n',
            '', content, flags=re.DOTALL)
        content = re.sub(
            r'  <group_state name="(?:open|close)" group="(?:left|right)_gripper">.*?</group_state>\n\n',
            '', content, flags=re.DOTALL)
        content = '\n'.join(
            line for line in content.split('\n')
            if not re.search(r'openarm_(?:left|right)_(?:hand"|left_finger|right_finger)', line)
        )
        # openarm_<side>_finger_joint1 ("motor 8") sits directly on the
        # left_arm/right_arm chain's base->hand_tcp path (link7 -> connector
        # -> ... -> hand_tcp - see amazing_hand_connector's joint comment in
        # openarm_robot.xacro), so without this it's a free variable to
        # KDL/OMPL for THAT group: dragging the interactive marker or any
        # left_arm Plan&Execute can drive it to an arbitrary/swinging value
        # chasing 6-DOF Cartesian goals with a now-8-DOF chain, independent
        # of whatever left_hand_rotate_controller last commanded - looks
        # like the joint "won't stop rotating" even though nothing is
        # touching it directly. passive_joint keeps FK correct (hand_tcp's
        # pose still reflects the joint's real position) while excluding it
        # from left_arm/right_arm's own IK/planning variables - it's only
        # ever actively planned/commanded through left_hand_rotate_controller
        # or a group that explicitly lists it (neither chain group does).
        # (Added via string injection, not left in the static SRDF, because
        # the line-strip above would delete a literal <passive_joint> tag
        # too - "finger_joint1" matches the same left_finger/right_finger
        # pattern being stripped for this ee_type.)
        content = content.replace(
            "  <!-- Virtual joints -->",
            '  <passive_joint name="openarm_left_finger_joint1"/>\n'
            '  <passive_joint name="openarm_right_finger_joint1"/>\n\n'
            "  <!-- Virtual joints -->"
        )
    elif ee_type == "openarm_hand":
        # Mirror image: strip the amazing_hand-only groups/group_states and
        # any disable_collisions line for its ahand_* links. The
        # end_effector group="left_gripper"/"right_gripper" attributes are
        # already correct in the static file for this direction - no
        # replace needed.
        content = re.sub(
            r'  <group name="(?:left|right)_hand_fingers">.*?</group>\n\n',
            '', content, flags=re.DOTALL)
        content = re.sub(
            r'  <group_state name="(?:open|close|home)" group="(?:left|right)_hand_fingers">.*?</group_state>\n\n',
            '', content, flags=re.DOTALL)
        content = '\n'.join(
            line for line in content.split('\n')
            if 'ahand' not in line
        )

    # body_type:=v2 adds an articulated neck_joint/head_joint (see
    # openarm_body.xacro); v1/v10 don't have them. Neither is in any static
    # SRDF group, so inject one here only when it actually exists in the
    # URDF - a static group would make v1 log "Joint
    # 'openarm_body_neck_joint' ... not known to the URDF" on every startup.
    if body_type == "v2":
        content = content.replace(
            "  <!-- Virtual joints -->",
            '  <group name="head">\n'
            '    <joint name="openarm_body_neck_joint"/>\n'
            '    <joint name="openarm_body_head_joint"/>\n'
            '  </group>\n\n'
            "  <!-- Virtual joints -->"
        )

    return content


# openarm_left_finger_joint1/openarm_right_finger_joint1 ("motor 8") is
# claimed by TWO mutually-exclusive controller entries in moveit_controllers.yaml
# depending on ee_type: left_gripper_controller/right_gripper_controller
# (openarm_hand's 2-finger gripper) or left_hand_rotate_controller/
# right_hand_rotate_controller (amazing_hand's connector rotation - see
# amazing_hand_connector's joint comment in openarm_robot.xacro). Only one
# pair is ever actually spawned by controller_manager for a given ee_type
# (moveit_bimanual.launch.py/bringup.launch.py's is_openarm_hand/
# is_amazing_hand spawner conditions); leaving both declared in move_group's
# own controller list lets MoveIt route a trajectory to whichever isn't
# actually running, since it picks by static joint-name match, not live
# controller state - "Action client not connected to action server" at
# execute time. Filtering here keeps this a config bug you configure away
# rather than a code path you debug every time ee_type changes.
_CONTROLLERS_BY_EE_TYPE = {
    "openarm_hand": ("left_gripper_controller", "right_gripper_controller"),
    "amazing_hand": ("left_hand_rotate_controller", "right_hand_rotate_controller"),
}


def load_moveit_controllers_for_ee_type(yaml_path: str, ee_type: str) -> dict:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    params = raw["/**"]["ros__parameters"]
    scm = params["moveit_simple_controller_manager"]

    keep = _CONTROLLERS_BY_EE_TYPE.get(ee_type)
    drop = _CONTROLLERS_BY_EE_TYPE.get(
        "amazing_hand" if ee_type == "openarm_hand" else "openarm_hand", ())
    if keep is not None:
        scm["controller_names"] = [n for n in scm["controller_names"] if n not in drop]
        for name in drop:
            scm.pop(name, None)

    return params
