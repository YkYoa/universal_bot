#pragma once

#include <sensor_msgs/msg/joint_state.hpp>
#include <string>
#include <vector>

namespace common
{
    /// Canonical joint names for one arm side, matching
    /// OpenArm_v10HW::generate_joint_names() (control/robot_hardware_interface/
    /// src/v10/hardware_interface.cpp) - the one place that actually drives the
    /// real motors. `side_prefix` is "left_" or "right_"; `count` is 7
    /// (openarm_hand/none) or 8 (amazing_hand's extra finger_joint1/"motor 8").
    ///
    /// This is the single place that convention is expressed for anything
    /// downstream of hardware_interface.cpp - callers pair values with these
    /// names instead of relying on a raw vector's length matching whichever
    /// ee_type happens to be live (see JointState-based joint_target/
    /// joint_sequence fields on ExecuteSkill.action).
    inline std::vector<std::string> armJointNames(const std::string & side_prefix, std::size_t count)
    {
        std::vector<std::string> names;
        for (std::size_t i = 1; i <= 7 && i <= count; ++i) {
            names.push_back("openarm_" + side_prefix + "joint" + std::to_string(i));
        }
        if (count == 8) {
            names.push_back("openarm_" + side_prefix + "finger_joint1");
        }
        return names;
    }

    /// "left_arm" -> "left_", "right_arm" -> "right_". Returns "" for
    /// anything else (e.g. "both_arms", which has no single side) - building
    /// names from an empty prefix yields names that match nothing on the live
    /// robot model, which fails loudly downstream instead of silently binding
    /// to the wrong side.
    inline std::string sidePrefixForGroup(const std::string & group)
    {
        if (group == "left_arm") return "left_";
        if (group == "right_arm") return "right_";
        return "";
    }

    /// Pairs `values` (7 or 8 joint positions) with the canonical names for
    /// `side_prefix` (see armJointNames). The receiving end resolves each
    /// name against whatever the live robot model actually has, so data
    /// authored for one ee_type still works for a robot booted under another -
    /// a name that doesn't exist there is simply skipped, not rejected.
    inline sensor_msgs::msg::JointState jointStateFor(
        const std::string & side_prefix, const std::vector<double> & values)
    {
        sensor_msgs::msg::JointState state;
        state.name = armJointNames(side_prefix, values.size());
        state.position = values;
        return state;
    }

    /// Appends `other`'s name/position pairs onto `state` - used to combine
    /// left_arm + right_arm data into one both_arms waypoint message.
    inline void appendJointState(
        sensor_msgs::msg::JointState & state, const sensor_msgs::msg::JointState & other)
    {
        state.name.insert(state.name.end(), other.name.begin(), other.name.end());
        state.position.insert(state.position.end(), other.position.begin(), other.position.end());
    }
} // namespace common
