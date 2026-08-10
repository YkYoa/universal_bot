#pragma once
// -----------------------------------------------------------------------------
// sequence_step.hpp
//
// The generic step model the FSM walks.
//
// This replaces the old fixed shape (one home step, then a body looped N
// times). A sequence is now an ordered list of steps of arbitrary type, which
// is what lets the Android app compose one by dragging and dropping. The old
// YAML shape still loads - YamlSequenceSource unrolls it into this list.
//
// The field set is flat and shared across every step type rather than a
// variant or a JSON blob held at runtime: a step carries maybe a dozen
// numbers, the type tag says which fields are meaningful, and keeping it a
// plain struct means the FSM's dispatch reads as one switch with no casting.
// The authoritative per-type field list is qvic_2026/qvic_2026/step_types.py,
// which is also what the API serves to the Android client.
// -----------------------------------------------------------------------------
#include <string>
#include <vector>

namespace sequence_executor {

// Values of Step::required_control_mode / SequenceSpec::required_control_mode.
// The arm's mode is fixed when the hardware initialises (the damiao register
// is written once during init), so these are validated before a run starts,
// never switched during one.
inline constexpr const char* kModeAny = "any";
inline constexpr const char* kModeMotion = "position|mit";
inline constexpr const char* kModeTorque = "torque";

struct Step
{
  int index = 0;
  std::string name;
  std::string type;                        // see step_types.py for the full list
  std::string required_control_mode = kModeAny;
  bool enabled = true;

  // Arm motion (move_joint, move_joint_sequence, move_pose, named_pose)
  std::string arm;                         // left_arm | right_arm | both_arms
  std::string waypoint;                    // "section/name"
  std::string right_waypoint;              // both_arms: appended after `waypoint`
  std::vector<double> positions;           // raw joints, alternative to waypoint
  std::string section;                     // move_joint_sequence
  std::string right_section;
  std::vector<int> exclude_points;         // 1-indexed
  double velocity = 0.0;                   // 0 = inherit from the sequence
  double acceleration = 0.0;

  // Cartesian (move_pose)
  std::vector<double> position;            // x, y, z
  std::vector<double> orientation;         // qx, qy, qz, qw
  bool position_only = false;
  std::string frame_id;

  // named_pose
  std::string group;
  std::string pose;

  // hand_pose - every vector present fires concurrently
  std::vector<double> left_yaw, left_flex, right_yaw, right_flex;
  double duration = 1.0;

  // gripper
  std::string side;                        // left | right
  std::string action;                      // open | close

  // wait, teach_hold
  double seconds = 0.0;

  // scene steps (add_object, attach_object, allow_collision, ...)
  std::string object_id;
  std::string link;
  std::string primitive;                   // box | sphere | cylinder
  std::vector<double> dimensions;
  std::vector<std::string> touch_links;
  std::vector<std::string> links;
};

struct SequenceSpec
{
  std::string name;
  std::string description;
  std::string arm = "left_arm";
  std::string planner_profile;
  std::string required_control_mode = kModeAny;
  int repeat = 1;                          // -1 = forever
  double velocity = 0.0;                   // 0 = use the planner profile's own
  double acceleration = 0.0;
  bool builtin = false;
  std::vector<Step> steps;
};

// Does a step needing `required` run on hardware that came up in `active`?
// An unknown active mode is permissive - a failed probe should not brick every
// sequence, it should just stop being a useful guard.
bool modeIsCompatible(const std::string& required, const std::string& active);

}  // namespace sequence_executor
