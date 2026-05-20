#pragma once

#include "behaviortree_cpp/action_node.h"
#include "moveit_msgs/msg/robot_trajectory.hpp"

namespace bt_executor {

class ExecuteTrajectory : public BT::SyncActionNode
{
public:
  ExecuteTrajectory(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<moveit_msgs::msg::RobotTrajectory>("trajectory", "The trajectory (unused, since plan executes it directly)"),
    };
  }

  BT::NodeStatus tick() override;
};

}  // namespace bt_executor
