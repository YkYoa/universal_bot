#pragma once
// -----------------------------------------------------------------------------
// skill_client.hpp
//
// Thin ExecuteSkill action-client wrapper - the same goal-field population
// bt_executor's PlanToJointTarget/PlanToJointSequence::setGoal() did
// (src/control/bt_executor/src/nodes/actions/plan_to_joint_{target,sequence}.cpp),
// called directly instead of through BT::RosActionNode::tick(). Event-driven:
// callers get a plain result callback, no polling/ticking.
// -----------------------------------------------------------------------------
#include <functional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <openarm_messages/action/execute_skill.hpp>

namespace sequence_executor {

class SkillClient
{
public:
  using ExecuteSkill = openarm_messages::action::ExecuteSkill;
  using ResultCallback = std::function<void(bool success, const std::string& error_message)>;

  explicit SkillClient(
    const rclcpp::Node::SharedPtr& node,
    const std::string& action_name = "robot_skills_server/execute_skill");

  void moveToJoint(
    const std::string& arm, const std::vector<double>& joint_targets, const std::string& planner_profile,
    double velocity_scaling, double acceleration_scaling, ResultCallback callback);

  void moveToJointSequence(
    const std::string& arm, const std::vector<double>& joint_sequence, const std::string& planner_profile,
    double velocity_scaling, double acceleration_scaling, ResultCallback callback);

  // Cancels whatever goal is in flight, if any. robot_skills' handle_cancel
  // calls TrajectoryExecutionManager::stopExecution, so this is what actually
  // brings a moving arm to a halt. The goal's result callback still fires,
  // with a non-SUCCEEDED code - the caller finds out through the same path as
  // any other failure. Returns false when there was nothing to cancel.
  bool cancelActiveGoal();

private:
  void sendGoal(ExecuteSkill::Goal goal, ResultCallback callback);

  rclcpp::Node::SharedPtr node_;
  rclcpp_action::Client<ExecuteSkill>::SharedPtr client_;
  rclcpp::Logger logger_;

  // Only ever one goal in flight: the FSM runs one step at a time.
  rclcpp_action::ClientGoalHandle<ExecuteSkill>::SharedPtr active_goal_;
};

}  // namespace sequence_executor
