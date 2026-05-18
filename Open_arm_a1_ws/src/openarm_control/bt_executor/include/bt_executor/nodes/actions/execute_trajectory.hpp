#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// execute_trajectory.hpp
//
// Async action node: sends a moveit_msgs::msg::RobotTrajectory to the
// /execute_trajectory action server.
//
// Inherits from BT::RosActionNode<moveit_msgs::action::ExecuteTrajectory>
//
// Input ports:
//   trajectory moveit_msgs::msg::RobotTrajectory
// ─────────────────────────────────────────────────────────────────────────────
#include "behaviortree_ros2/bt_action_node.hpp"
#include "moveit_msgs/action/execute_trajectory.hpp"
#include "moveit_msgs/msg/robot_trajectory.hpp"

namespace bt_executor {

class ExecuteTrajectory
  : public BT::RosActionNode<moveit_msgs::action::ExecuteTrajectory>
{
public:
  using ExecuteTrajAction = moveit_msgs::action::ExecuteTrajectory;

  ExecuteTrajectory(const std::string & name,
                    const BT::NodeConfig & config,
                    const BT::RosNodeParams & params)
  : BT::RosActionNode<ExecuteTrajAction>(name, config, params) {}

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<moveit_msgs::msg::RobotTrajectory>("trajectory", "The trajectory to execute"),
    });
  }

  bool setGoal(Goal & goal) override;
  BT::NodeStatus onResultReceived(const WrappedResult & result) override;
  BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override;
};

}  // namespace bt_executor
