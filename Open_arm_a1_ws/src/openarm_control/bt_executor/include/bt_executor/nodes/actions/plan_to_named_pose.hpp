#pragma once

#include "behaviortree_ros2/bt_action_node.hpp"
#include "openarm_messages/action/execute_skill.hpp"
#include "moveit_msgs/msg/robot_trajectory.hpp"
#include "bt_executor/blackboard_keys.hpp"
#include <string>

namespace bt_executor {

class PlanToNamedPose
  : public BT::RosActionNode<openarm_messages::action::ExecuteSkill>
{
public:
  using ExecuteSkillAction = openarm_messages::action::ExecuteSkill;

  PlanToNamedPose(const std::string & name,
                  const BT::NodeConfig & config,
                  const BT::RosNodeParams & params)
  : BT::RosActionNode<ExecuteSkillAction>(name, config, params) {}

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("arm", "left_arm | right_arm"),
      BT::InputPort<std::string>("pose_name", "home | ready"),
      BT::InputPort<double>("duration", 3.0, "Trajectory duration [s]"),
      BT::OutputPort<moveit_msgs::msg::RobotTrajectory>("output_trajectory", "Dummy trajectory for compat"),
    });
  }

  bool setGoal(Goal & goal) override;
  BT::NodeStatus onResultReceived(const WrappedResult & result) override;
  BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override;
  
  BT::NodeStatus onFeedback(const std::shared_ptr<const Feedback> /*feedback*/) override
  {
    return BT::NodeStatus::RUNNING;
  }
};

}  // namespace bt_executor
