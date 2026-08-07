#pragma once

// Thin topic-based rclcpp::Node wrapper around
// amazing_hand_kinematics::HandSolver - kept as a standalone dev/debug tool,
// same topics/behavior as openarm_description/scripts/hand_kinematics_node.py.
//
// The actual ros2_control integration is
// control/robot_hardware_interface's AmazingHandHW plugin, which calls
// HandSolver directly (no topics, no separate node) from read().

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "amazing_hand_kinematics/amazing_hand_kinematics.hpp"

namespace robot_control
{

class HandKinematicNode : public rclcpp::Node
{
public:
  explicit HandKinematicNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  using JointStateMsg = sensor_msgs::msg::JointState;

  void onJointStates(const JointStateMsg& msg);
  void republishStates();

  std::unique_ptr<amazing_hand_kinematics::HandSolver> solver_;
  std::optional<std::pair<std::vector<std::string>, std::vector<double>>> last_full_state_;

  rclcpp::Publisher<JointStateMsg>::SharedPtr pub_;
  rclcpp::Publisher<JointStateMsg>::SharedPtr alias_pub_;
  rclcpp::Subscription<JointStateMsg>::SharedPtr sub_;
  rclcpp::Subscription<JointStateMsg>::SharedPtr cmd_sub_;
  rclcpp::TimerBase::SharedPtr state_timer_;
};

}  // namespace robot_control
