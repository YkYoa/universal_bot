#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// control_gripper.hpp
//
// Async action nodes for opening and closing the end-effector.
// Uses FollowJointTrajectory rather than MoveGroup for deterministic,
// fast 1-DOF jaw control.
//
// XML usage:
//   <CloseGripper  arm="{plan_active_arm}" close_position="0.0" />
//   <OpenGripper arm="{plan_active_arm}" open_position="0.044" />
// ─────────────────────────────────────────────────────────────────────────────
#include "behaviortree_ros2/bt_action_node.hpp"
#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "bt_executor/blackboard_keys.hpp"

namespace bt_executor {

// ── CloseGripper ───────────────────────────────────────────────────────────────
class CloseGripper
  : public BT::RosActionNode<control_msgs::action::FollowJointTrajectory>
{
public:
  using FJT = control_msgs::action::FollowJointTrajectory;

  CloseGripper(const std::string & name,
               const BT::NodeConfig & config,
               const BT::RosNodeParams & params)
  : BT::RosActionNode<FJT>(name, config, params) {}

  BT::NodeStatus tick() override
  {
    std::string arm;
    getInput("arm", arm);
    std::string target_action = (arm == "right_arm")
      ? "right_gripper_controller/follow_joint_trajectory"
      : "left_gripper_controller/follow_joint_trajectory";
    setActionName(target_action);
    return BT::RosActionNode<FJT>::tick();
  }

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("arm", "left_arm | right_arm"),
      BT::InputPort<double>("close_position", 0.0,
        "Finger joint position when closed [m]. 0.0 = fully closed."),
      BT::InputPort<double>("duration", 1.5, "Gripper close time [s]"),
    });
  }

  bool setGoal(Goal & goal) override
  {
    std::string arm;
    double close_pos = 0.0, duration = 1.5;
    getInput("arm",            arm);
    getInput("close_position", close_pos);
    getInput("duration",       duration);

    const std::string joint = (arm == "left_arm")
      ? "openarm_left_finger_joint1"
      : "openarm_right_finger_joint1";
    goal.trajectory.joint_names = {joint};

    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions = {close_pos};
    pt.time_from_start.sec = static_cast<int32_t>(duration);
    pt.time_from_start.nanosec =
      static_cast<uint32_t>((duration - std::floor(duration)) * 1e9);
    goal.trajectory.points.push_back(pt);

    RCLCPP_INFO(logger(), "[CloseGripper] %s gripper → %.3f m",
      arm.c_str(), close_pos);
    return true;
  }

  BT::NodeStatus onResultReceived(const WrappedResult & result) override
  {
    if (result.result->error_code == FJT::Result::SUCCESSFUL) {
      config().blackboard->set(BB_STATUS_MSG, std::string("grasp: success"));
      return BT::NodeStatus::SUCCESS;
    }
    RCLCPP_WARN(logger(), "[CloseGripper] failed error_code=%d",
      result.result->error_code);
    return BT::NodeStatus::FAILURE;
  }

  BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override
  {
    RCLCPP_ERROR(logger(), "[CloseGripper] action failed: %s",
      BT::toStr(error));
    return BT::NodeStatus::FAILURE;
  }
};

// ── OpenGripper ─────────────────────────────────────────────────────────────
class OpenGripper
  : public BT::RosActionNode<control_msgs::action::FollowJointTrajectory>
{
public:
  using FJT = control_msgs::action::FollowJointTrajectory;

  OpenGripper(const std::string & name,
                const BT::NodeConfig & config,
                const BT::RosNodeParams & params)
  : BT::RosActionNode<FJT>(name, config, params) {}

  BT::NodeStatus tick() override
  {
    std::string arm;
    getInput("arm", arm);
    std::string target_action = (arm == "right_arm")
      ? "right_gripper_controller/follow_joint_trajectory"
      : "left_gripper_controller/follow_joint_trajectory";
    setActionName(target_action);
    return BT::RosActionNode<FJT>::tick();
  }

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("arm", "left_arm | right_arm"),
      BT::InputPort<double>("open_position", 0.044,
        "Finger position when open [m]. 0.044 = fully open."),
      BT::InputPort<double>("duration", 1.0, "duration in seconds"),
    });
  }

  bool setGoal(Goal & goal) override
  {
    std::string arm;
    double open_pos = 0.044, duration = 1.0;
    getInput("arm",           arm);
    getInput("open_position", open_pos);
    getInput("duration",      duration);

    const std::string joint = (arm == "left_arm")
      ? "openarm_left_finger_joint1"
      : "openarm_right_finger_joint1";
    goal.trajectory.joint_names = {joint};

    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions = {open_pos};
    pt.time_from_start.sec = static_cast<int32_t>(duration);
    pt.time_from_start.nanosec =
      static_cast<uint32_t>((duration - std::floor(duration)) * 1e9);
    goal.trajectory.points.push_back(pt);

    RCLCPP_INFO(logger(), "[OpenGripper] %s gripper → %.3f m",
      arm.c_str(), open_pos);
    return true;
  }

  BT::NodeStatus onResultReceived(const WrappedResult & result) override
  {
    if (result.result->error_code == FJT::Result::SUCCESSFUL) {
      config().blackboard->set(BB_STATUS_MSG, std::string("release: success"));
      return BT::NodeStatus::SUCCESS;
    }
    return BT::NodeStatus::FAILURE;
  }

  BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override
  {
    RCLCPP_ERROR(logger(), "[OpenGripper] action failed: %s",
      BT::toStr(error));
    return BT::NodeStatus::FAILURE;
  }
};

}  // namespace bt_executor
