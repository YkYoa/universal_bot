#include "bt_executor/nodes/actions/plan_to_named_pose.hpp"
#include "rclcpp/rclcpp.hpp"

namespace bt_executor {

bool PlanToNamedPose::setGoal(Goal & goal)
{
  std::string arm, pose_name;
  if (!getInput("arm", arm)) {
    RCLCPP_ERROR(logger(), "[PlanToNamedPose] arm port not set");
    return false;
  }
  if (!getInput("pose_name", pose_name)) {
    RCLCPP_ERROR(logger(), "[PlanToNamedPose] pose_name port not set");
    return false;
  }

  // Read config from blackboard
  std::string profile_name = "";
  std::string planning_mode = "normal";

  (void)config().blackboard->get(BB_PLANNER_PROFILE, profile_name);
  (void)config().blackboard->get(BB_PLANNING_MODE, planning_mode);

  // Set ExecuteSkill goal fields
  goal.skill_name = "move_to_named_pose";
  goal.arm = arm;
  goal.planner_profile = profile_name;
  goal.planning_mode = planning_mode;
  goal.named_pose = pose_name;
  goal.velocity_override = 0.0; // default profile speed
  goal.position_only = false;

  RCLCPP_INFO(logger(), 
    "[PlanToNamedPose] ExecuteSkill: %s | arm: %s | profile: %s | mode: %s | named_pose: %s",
    goal.skill_name.c_str(), goal.arm.c_str(), goal.planner_profile.c_str(), 
    goal.planning_mode.c_str(), pose_name.c_str());

  return true;
}

BT::NodeStatus PlanToNamedPose::onResultReceived(const WrappedResult & result)
{
  if (result.result->success) {
    config().blackboard->set(BB_STATUS_MSG, std::string("plan_to_named_pose: success"));
    
    // Save a dummy trajectory to output port for compatibility with existing BT trees
    moveit_msgs::msg::RobotTrajectory dummy_traj;
    setOutput("output_trajectory", dummy_traj);

    return BT::NodeStatus::SUCCESS;
  }

  const std::string msg = std::string("plan_to_named_pose skill failed: ") + result.result->error_message;
  config().blackboard->set(BB_STATUS_MSG, msg);
  config().blackboard->set(BB_REPLAN_NEEDED, true);
  RCLCPP_WARN(logger(), "[PlanToNamedPose] %s", msg.c_str());
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus PlanToNamedPose::onFailure(BT::ActionNodeErrorCode error)
{
  RCLCPP_ERROR(logger(), "[PlanToNamedPose] action failed: %s", BT::toStr(error));
  config().blackboard->set(BB_REPLAN_NEEDED, true);
  return BT::NodeStatus::FAILURE;
}

}  // namespace bt_executor
