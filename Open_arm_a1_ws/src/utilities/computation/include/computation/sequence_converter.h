#pragma once

#include <string>
#include <vector>

namespace utilities {
namespace computation {

/**
 * @brief Convert an OpenArm sequence YAML config file into a BehaviorTree.CPP v4 XML file.
 * 
 * @param yaml_path Path to the input sequence.yaml file
 * @param out_path Path to the output behavior tree XML file
 * @param section_filter Optional name of specific section to convert (e.g. "homePoses")
 * @param arm_override Optional string to override the MoveIt planning group/arm for all nodes
 */
void convertSequenceToBt(const std::string& yaml_path,
                         const std::string& out_path,
                         const std::string& section_filter = "",
                         const std::string& arm_override = "");

/**
 * @brief Convert two matching per-arm sections of a sequence YAML (e.g.
 * "waveEllipse" / "waveEllipseR", same waypoint count, recorded from
 * left_arm/right_arm respectively) into ONE BehaviorTree.CPP v4 XML file
 * targeting the SRDF "both_arms" composite group - a single
 * PlanToJointSequence with each waypoint's left-arm values immediately
 * followed by that waypoint's right-arm values (left_arm-then-right_arm is
 * both_arms' declared SRDF joint order). This is what makes the two arms
 * move as one blended trajectory instead of two independently-timed ones.
 *
 * @param yaml_path Path to the input sequence.yaml file
 * @param out_path Path to the output behavior tree XML file
 * @param left_section Section name providing the left_arm waypoints
 * @param right_section Section name providing the right_arm waypoints (must
 *        have the same waypoint count as left_section)
 * @param profile Planner profile for the emitted <SetPlannerConfig/> -
 *        must be an OMPL profile (e.g. "realtime_rrt"/"safe_rrt"), not a
 *        Pilz one: Pilz refuses to plan for any group without a
 *        kinematics.yaml IK-solver entry, which both_arms structurally
 *        can't have (two independent chains, not one KDL-solvable chain).
 * @param tree_name BehaviorTree ID / root main_tree_to_execute name
 * @param home_section Optional section name providing a fixed starting pose,
 *        emitted as a PlanToJointTarget("both_arms") + hand-hold pose BEFORE
 *        the wave, so the trajectory always starts from a known state.
 *        Expected keys (7 values each for arms, 4 each for hand):
 *        laHomeAngle, raHomeAngle (required if home_section is set),
 *        lhHomeYaw, lhHomeFlex, rhHomeYaw, rhHomeFlex (optional, amazing_hand
 *        alias joints - include a side's Yaw+Flex pair to hold that hand's
 *        pose for the whole sequence via SetHandYaw/SetHandFlex).
 * @param repeat_cycles If not 1, wraps ONLY the wave (SetPlannerConfig +
 *        PlanToJointSequence) in a <Repeat num_cycles="repeat_cycles">
 *        decorator - home_section (if given) stays outside the loop and so
 *        runs exactly once. -1 means repeat forever (BT.CPP convention).
 * @param exclude_points 1-indexed point numbers to drop from the interleaved
 *        sequence (e.g. a point known to self-collide when left+right are
 *        combined - see check_bimanual_collision). Both per-arm sections
 *        stay untouched in the YAML; only this generated XML skips them.
 */
void convertBothArmsSequenceToBt(const std::string& yaml_path,
                                  const std::string& out_path,
                                  const std::string& left_section,
                                  const std::string& right_section,
                                  const std::string& profile = "realtime_rrt",
                                  const std::string& tree_name = "Both",
                                  const std::string& home_section = "",
                                  int repeat_cycles = 1,
                                  const std::vector<int>& exclude_points = {});

} // namespace computation
} // namespace utilities
    