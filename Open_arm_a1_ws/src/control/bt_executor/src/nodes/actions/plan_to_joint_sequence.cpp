#include "bt_executor/nodes/actions/plan_to_joint_sequence.hpp"
#include "rclcpp/rclcpp.hpp"

namespace bt_executor {

bool PlanToJointSequence::setGoal(Goal & goal)
{
  std::string arm;
  std::vector<double> joint_sequence;
  if (!getInput("arm", arm)) {
    RCLCPP_ERROR(logger(), "[PlanToJointSequence] arm port not set");
    return false;
  }
  if (!getInput("joint_sequence", joint_sequence)) {
    RCLCPP_ERROR(logger(), "[PlanToJointSequence] joint_sequence port not set");
    return false;
  }
  if (joint_sequence.empty()) {
    RCLCPP_ERROR(logger(), "[PlanToJointSequence] joint_sequence must not be empty");
    return false;
  }

  // Read config from blackboard
  std::string profile_name = "";
  std::string planning_mode = "normal";

  (void)config().blackboard->get(BB_PLANNER_PROFILE, profile_name);
  (void)config().blackboard->get(BB_PLANNING_MODE, planning_mode);

  double velocity_scaling = 0.0;
  getInput("velocity_scaling", velocity_scaling);
  double acceleration_scaling = 0.0;
  getInput("acceleration_scaling", acceleration_scaling);

  // Set ExecuteSkill goal fields
  goal.skill_name = "move_to_joint_sequence";
  goal.arm = arm;
  goal.planner_profile = profile_name;
  goal.planning_mode = planning_mode;
  goal.joint_sequence = joint_sequence;
  goal.velocity_override = velocity_scaling; // 0.0 = skill's default profile speed
  goal.acceleration_override = acceleration_scaling; // 0.0 = skill's default profile
  goal.position_only = false;

  RCLCPP_INFO(logger(),
    "[PlanToJointSequence] ExecuteSkill: %s | arm: %s | profile: %s | mode: %s | velocity_scaling: %.2f | "
    "acceleration_scaling: %.2f | joint_sequence values: %zu",
    goal.skill_name.c_str(), goal.arm.c_str(), goal.planner_profile.c_str(),
    goal.planning_mode.c_str(), velocity_scaling, acceleration_scaling, joint_sequence.size());

  return true;
}

BT::NodeStatus PlanToJointSequence::onResultReceived(const WrappedResult & result)
{
  if (result.result->success) {
    config().blackboard->set(BB_STATUS_MSG, std::string("plan_to_joint_sequence: success"));

    // Save a dummy trajectory to output port for compatibility with existing BT trees
    moveit_msgs::msg::RobotTrajectory dummy_traj;
    setOutput("output_trajectory", dummy_traj);

    return BT::NodeStatus::SUCCESS;
  }

  const std::string msg = std::string("plan_to_joint_sequence skill failed: ") + result.result->error_message;
  config().blackboard->set(BB_STATUS_MSG, msg);
  config().blackboard->set(BB_REPLAN_NEEDED, true);
  RCLCPP_WARN(logger(), "[PlanToJointSequence] %s", msg.c_str());
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus PlanToJointSequence::onFailure(BT::ActionNodeErrorCode error)
{
  RCLCPP_ERROR(logger(), "[PlanToJointSequence] action failed: %s", BT::toStr(error));
  config().blackboard->set(BB_REPLAN_NEEDED, true);
  return BT::NodeStatus::FAILURE;
}

}  // namespace bt_executor
