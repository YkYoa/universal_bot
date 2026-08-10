#pragma once
// -----------------------------------------------------------------------------
// sequence_interpreter.hpp
//
// The state machine itself. Fixed-shape (not a generic step dispatcher),
// matching sequence.yaml's flat `sequences:` schema:
//   1. if home_section set: run its arm step (once) then its hand-hold step
//      (once, parallel yaw+flex per side present).
//   2. loop the body `repeat` times (-1 = forever): either one interleaved
//      joint_sequence call (body_section[+body_right_section]), or one pass
//      through body_sections in order (one hand-pose step per section).
//   3. any ROS call failing halts the sequence and publishes on ~/fault -
//      no silent continue, no crash, no more looping.
//
// Event-driven: advances on SkillClient/HandGripperClient result callbacks,
// not a tick loop.
// -----------------------------------------------------------------------------
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "sequence_executor/sequence_yaml.hpp"
#include "sequence_executor/skill_client.hpp"
#include "sequence_executor/hand_gripper_client.hpp"

namespace sequence_executor {

class SequenceInterpreter
{
public:
  SequenceInterpreter(
    rclcpp::Node::SharedPtr node, std::shared_ptr<SequenceYaml> yaml, std::shared_ptr<SkillClient> skill_client,
    std::shared_ptr<HandGripperClient> hand_client);

  // Starts `sequence_name` running: home (if any) once, then body x repeat
  // (or forever). Returns immediately - all execution happens via ROS
  // callbacks on this node's executor.
  void run(const std::string& sequence_name);

private:
  void runHome(const SequenceDef& def, std::function<void()> on_done);

  // Scans `section` for lh*/rh* Yaw+Flex pairs and fires whichever sides
  // are present concurrently, calling on_done(true) once all complete
  // successfully (or fail()+on_done(false) on any failure) - shared by both
  // home_section's one-time hand hold and each body_sections step.
  void runHandPoseSection(const std::string& section, double duration, std::function<void(bool ok)> on_done);

  void loopBody(SequenceDef def, int remaining);
  void runBodyOnce(const SequenceDef& def, std::function<void(bool ok)> on_done);
  void runBodySectionsStep(SequenceDef def, std::size_t index, std::function<void(bool ok)> on_done);

  void publishStatus(const std::string& status);
  void fail(const std::string& reason);

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<SequenceYaml> yaml_;
  std::shared_ptr<SkillClient> skill_client_;
  std::shared_ptr<HandGripperClient> hand_client_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fault_pub_;
  rclcpp::Logger logger_;
  std::string sequence_name_;
};

}  // namespace sequence_executor
