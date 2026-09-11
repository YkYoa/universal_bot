#pragma once
// -----------------------------------------------------------------------------
// hand_gripper_client.hpp
//
// FollowJointTrajectory goal-builders for amazing_hand's yaw/flex alias
// joints and the 1-DOF parallel gripper - ported near-verbatim from
// bt_executor's control_hand.hpp (SetHandYaw/SetHandFlex) and
// control_gripper.hpp (OpenGripper/CloseGripper), called directly instead
// of through BT::RosActionNode::tick(). One rclcpp_action::Client per
// controller, created lazily on first use per side (mirrors those nodes'
// per-tick setActionName() side dispatch).
// -----------------------------------------------------------------------------
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>

namespace sequence_executor {

/// See file header comment: FollowJointTrajectory goal-builders for
/// amazing_hand's yaw/flex joints, the parallel gripper, and the head.
class HandGripperClient
{
public:
  using FJT = control_msgs::action::FollowJointTrajectory;
  using ResultCallback = std::function<void(bool success, const std::string& error_message)>;

  /// Stores `node`; action clients are created lazily per controller (see clientFor()).
  explicit HandGripperClient(const rclcpp::Node::SharedPtr& node);

  // arm: "left_arm" | "right_arm". positions: exactly 4 values (fingers
  // 1-4, 4=thumb), matching SetHandYaw/SetHandFlex's XML port shape.
  /// Sends yaw (abduction) targets for `arm`'s 4 fingers.
  void setHandYaw(
    const std::string& arm, const std::vector<double>& positions, double duration, ResultCallback callback);
  /// Sends flex targets for `arm`'s 4 fingers.
  void setHandFlex(
    const std::string& arm, const std::vector<double>& positions, double duration, ResultCallback callback);

  /// Sends `open_position` to `arm`'s parallel gripper.
  void openGripper(const std::string& arm, double open_position, double duration, ResultCallback callback);
  /// Sends `close_position` to `arm`'s parallel gripper.
  void closeGripper(const std::string& arm, double close_position, double duration, ResultCallback callback);

  /// Neck pan and head tilt, in radians, through head_controller. Separate from
  /// the hands only by which controller it targets - the trajectory shape is
  /// the same, so it shares the machinery below.
  void setHead(double pan, double tilt, double duration, ResultCallback callback);

  /// Any joint list on any FollowJointTrajectory controller. The composite
  /// "move several groups at once" step needs this because the group it drives
  /// is chosen at runtime, not at compile time.
  void sendTrajectory(
    const std::string& action_name, const std::vector<std::string>& joint_names,
    const std::vector<double>& positions, double duration, ResultCallback callback);

private:
  /// Returns (creating if needed) the FollowJointTrajectory client for `action_name`.
  rclcpp_action::Client<FJT>::SharedPtr clientFor(const std::string& action_name);
  /// Builds and sends a one-joint, one-point trajectory goal.
  void sendSingleJointGoal(
    const std::string& action_name, const std::string& joint_name, double position, double duration,
    ResultCallback callback);
  /// Builds and sends a four-joint, one-point trajectory goal (the amazing_hand
  /// yaw/flex shape).
  void sendFourJointGoal(
    const std::string& action_name, const std::vector<std::string>& joint_names,
    const std::vector<double>& positions, double duration, ResultCallback callback);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_;
  std::map<std::string, rclcpp_action::Client<FJT>::SharedPtr> clients_;
};

}  // namespace sequence_executor
