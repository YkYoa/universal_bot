#pragma once

#include "behaviortree_ros2/bt_action_node.hpp"
#include "openarm_messages/action/execute_skill.hpp"
#include "moveit_msgs/msg/robot_trajectory.hpp"
#include "bt_executor/blackboard_keys.hpp"
#include <string>
#include <vector>

namespace bt_executor {

/**
 * @brief Plans/executes a list of joint-space waypoints as ONE continuous
 *        blended trajectory via the "move_to_joint_sequence" skill
 *        (robot_skills/MoveToJointSequenceSkill) - unlike PlanToJointTarget,
 *        which is one independent plan+execute+full-stop per waypoint.
 *        Emitted by sequence_converter.cpp for a run of consecutive
 *        "<arm><Section><N>Angle" auto-numbered keys.
 */
class PlanToJointSequence
  : public BT::RosActionNode<openarm_messages::action::ExecuteSkill>
{
public:
  using ExecuteSkillAction = openarm_messages::action::ExecuteSkill;

  PlanToJointSequence(const std::string & name,
                  const BT::NodeConfig & config,
                  const BT::RosNodeParams & params)
  : BT::RosActionNode<ExecuteSkillAction>(name, config, params) {}

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("arm", "left_arm | right_arm"),
      BT::InputPort<std::vector<double>>(
        "joint_sequence", "Semicolon-separated joint angles [rad], flat stride-7 chunks (N waypoints x 7 joints)"),
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
