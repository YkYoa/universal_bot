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
