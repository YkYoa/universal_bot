#pragma once

#include "behaviortree_ros2/bt_action_node.hpp"
#include "openarm_messages/action/execute_skill.hpp"
#include "moveit_msgs/msg/robot_trajectory.hpp"
#include "bt_executor/blackboard_keys.hpp"
#include <string>
#include <vector>

namespace bt_executor {

/**
 * @brief Plans/executes to a raw joint-space target via the "move_to_joint"
 *        skill (robot_skills/MoveToJointSkill) - no SRDF named group_state
 *        required, unlike PlanToNamedPose. Emitted by sequence_converter.cpp
 *        for sequence.yaml's "*Angle" keys.
 */
class PlanToJointTarget
  : public BT::RosActionNode<openarm_messages::action::ExecuteSkill>
{
public:
  using ExecuteSkillAction = openarm_messages::action::ExecuteSkill;

  PlanToJointTarget(const std::string & name,
                  const BT::NodeConfig & config,
                  const BT::RosNodeParams & params)
  : BT::RosActionNode<ExecuteSkillAction>(name, config, params) {}

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("arm", "left_arm | right_arm"),
      BT::InputPort<std::vector<double>>("joint_targets", "Semicolon-separated joint angles [rad]"),
      BT::InputPort<double>(
        "velocity_scaling", 0.0, "MoveIt velocity scaling factor [0-1]; 0 = skill's default profile speed"),
      BT::InputPort<double>(
        "acceleration_scaling", 0.0, "MoveIt acceleration scaling factor [0-1]; 0 = skill's default profile"),
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
