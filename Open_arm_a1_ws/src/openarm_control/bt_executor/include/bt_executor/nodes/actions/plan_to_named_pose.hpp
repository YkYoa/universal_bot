#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// plan_to_named_pose.hpp
//
// Synchronous action node: generates a joint-space trajectory (by name)
// and outputs it as a RobotTrajectory.
//
// Inherits from BT::SyncActionNode
//
// Input ports:
//   arm              "left_arm" | "right_arm"
//   pose_name        std::string (the SRDF state name)
//   duration         double (Trajectory duration [s])
//
// Output ports:
//   output_trajectory moveit_msgs::msg::RobotTrajectory
// ─────────────────────────────────────────────────────────────────────────────
#include "behaviortree_cpp/action_node.h"
#include "moveit_msgs/msg/robot_trajectory.hpp"
#include "rclcpp/rclcpp.hpp"
#include "bt_executor/blackboard_keys.hpp"
#include <map>
#include <vector>
#include <string>

namespace bt_executor {

class PlanToNamedPose : public BT::SyncActionNode
{
public:
  PlanToNamedPose(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::string>("arm", "left_arm | right_arm"),
      BT::InputPort<std::string>("pose_name", "home | ready"),
      BT::InputPort<double>("duration", 3.0, "Trajectory duration [s]"),
      BT::OutputPort<moveit_msgs::msg::RobotTrajectory>("output_trajectory", "The planned trajectory"),
    };
  }

  BT::NodeStatus tick() override;

private:
  // Named pose joint values — mirrors the SRDF group_states
  static const std::map<std::string, std::map<std::string, std::vector<double>>>
    named_poses_;
};

}  // namespace bt_executor
