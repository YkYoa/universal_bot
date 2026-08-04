#pragma once

#include "behaviortree_ros2/bt_action_node.hpp"
#include "openarm_messages/action/execute_skill.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "moveit_msgs/msg/robot_trajectory.hpp"
#include "bt_executor/blackboard_keys.hpp"

namespace bt_executor {

class PlanToPose
  : public BT::RosActionNode<openarm_messages::action::ExecuteSkill>
{
public:
  using ExecuteSkillAction = openarm_messages::action::ExecuteSkill;

  PlanToPose(const std::string & name,
             const BT::NodeConfig & config,
             const BT::RosNodeParams & params)
  : BT::RosActionNode<ExecuteSkillAction>(name, config, params) {}

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("arm", "left_arm | right_arm"),
      BT::InputPort<geometry_msgs::msg::PoseStamped>("target_pose"),
      BT::InputPort<double>("velocity_scaling", 0.3, "Velocity scaling"),
      BT::InputPort<double>("acceleration_scaling", 0.0, "Acceleration scaling [0-1]; 0 = skill's default profile"),
      BT::InputPort<bool>("position_only", false, "If true, skip orientation constraint"),
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
