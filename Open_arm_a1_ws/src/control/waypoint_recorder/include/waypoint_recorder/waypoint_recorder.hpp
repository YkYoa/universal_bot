#pragma once

#include <map>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

namespace waypoint_recorder
{

// "la"/"ra" (arm) -> URDF joint-name side prefix ("left"/"right"). Gripper
// prefixes are listed for completeness but rejected by WaypointRecorder -
// gripper waypoint recording isn't supported (single joint, no arm chain).
extern const std::map<std::string, std::string> kArmPrefixToSide;

std::string toPascalCase(const std::string& s);
std::string trim(const std::string& s);

// Inserts or updates `  key: value` under `section:` in the yaml file at
// file_path. Does NOT round-trip through a yaml parser/emitter - that would
// discard every comment (no ROS/common C++ yaml library preserves comments
// on write). Instead does a targeted text insertion: finds the section
// header line, finds where that section's indented block ends, splices the
// new (or updated) key line in, leaves every other line untouched.
// Returns true on success (false + stderr message on I/O failure).
bool writeWaypoint(
  const std::string& file_path, const std::string& section, const std::string& key, const std::string& value);

// Captures the arm's current joint positions from /joint_states and records
// them as a "<arm_prefix><PascalName>Angle" waypoint into a sequence.yaml.
// One instance per recording session: holds its own subscription + executor
// so repeated recordOne() calls each observe a fresh message, not a stale
// one from a previous call.
class WaypointRecorder
{
public:
  explicit WaypointRecorder(rclcpp::Node::SharedPtr node, double timeout_sec = 5.0);

  // Records one waypoint. arm_prefix must be "la" or "ra". On failure,
  // returns false and out_error is set (subscription timeout, missing joint
  // in /joint_states, or yaml write failure).
  bool recordOne(
    const std::string& arm_prefix, const std::string& section, const std::string& waypoint_name,
    const std::string& file_path, std::string& out_error);

  // Records both arms from a single /joint_states capture - writes both
  // "la<Name>Angle" and "ra<Name>Angle" from the exact same instant, for
  // bimanual poses where both arms need to match up. On failure returns
  // false with out_error set (either side's error, or both joined).
  bool recordBoth(
    const std::string& section, const std::string& waypoint_name, const std::string& file_path,
    std::string& out_error);

  // Interactive loop over stdin, three phases:
  //  - Arm-select: prompts for 'la' or 'ra'. Empty line here quits.
  //  - Section-select: lists sections already in file_path, prompts for one
  //    to record into (existing name reuses it, new name creates it). Empty
  //    line here quits.
  //  - Pose-recording: previews the next auto-numbered
  //    "<arm><PascalCase(section)><N>Angle" key; plain Enter records it as
  //    -is, a typed name overrides it, 'a' goes back to arm-select (keeping
  //    the current section - handy for recording the same pose set on both
  //    arms), 's' goes back to section-select, 'q' quits.
  // default_arm/default_section (either may be empty) skip their respective
  // first prompt. One session can cover both arms and several sections.
  // Individual recordOne() failures are printed to stderr and the loop
  // continues (a bad capture shouldn't lose the rest of the session).
  // Returns the count successfully recorded.
  int recordLoop(const std::string& default_arm, const std::string& default_section, const std::string& file_path);

private:
  // Blocks (spinning the internal executor) until a /joint_states message
  // newer than the previous capture arrives, or timeout_sec_ elapses.
  // Returns nullptr on timeout.
  sensor_msgs::msg::JointState::SharedPtr captureJointState();

  rclcpp::Node::SharedPtr node_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
  sensor_msgs::msg::JointState::SharedPtr latest_msg_;
  bool got_new_{false};
  double timeout_sec_;
};

}  // namespace waypoint_recorder
