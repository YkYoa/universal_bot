#pragma once

#include <vector>

namespace common
{
    /// Adapts a joint vector to a planning group's actual variable count.
    ///
    /// left_arm/right_arm's live DOF count depends on which ee_type built the
    /// robot's SRDF: 7 under openarm_hand/none, 8 under amazing_hand (the 8th
    /// being openarm_<side>_finger_joint1, "motor 8" - see
    /// moveit_cpp_planner_manager.cpp's checkJointVectorSize). Waypoint data
    /// recorded for one ee_type therefore often needs a one-value adjustment
    /// to play back correctly under the other.
    ///
    /// Only a difference of exactly one is adapted:
    ///  - `values.size() + 1 == target_count`: pad a trailing 0.0. This
    ///    matches the convention already used by every 8-value waypoint in
    ///    sequence.yaml (homePoses, waveEllipse, waveEllipseR all end their
    ///    amazing_hand entries in ", 0.0") - not a new guess, the same default
    ///    those already-deployed waypoints use for that joint.
    ///  - `values.size() == target_count + 1`: drop the trailing value. This
    ///    mirrors how waveEllipseOpenarm/waveEllipseROpenarm were hand-derived
    ///    from waveEllipse/waveEllipseR.
    ///
    /// Any other difference is left untouched and returned as-is - that is a
    /// real data error (e.g. 6 or 9 values), not an ee_type mismatch, and must
    /// still fail loudly at the caller's own group-size check instead of being
    /// silently "fixed" here.
    inline std::vector<double> adaptJointVectorToGroupSize(
        const std::vector<double> & values, std::size_t target_count)
    {
        if (values.size() == target_count) {
            return values;
        }
        if (values.size() + 1 == target_count) {
            std::vector<double> padded = values;
            padded.push_back(0.0);
            return padded;
        }
        if (values.size() == target_count + 1) {
            return std::vector<double>(values.begin(), values.end() - 1);
        }
        return values;
    }
} // namespace common
