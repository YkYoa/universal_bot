#include "sequence_executor/skill_client.hpp"

namespace sequence_executor {

SkillClient::SkillClient(const rclcpp::Node::SharedPtr& node, const std::string& action_name)
: node_(node), logger_(rclcpp::get_logger("SkillClient"))
{
  client_ = rclcpp_action::create_client<ExecuteSkill>(node_, action_name);
}

void SkillClient::moveToJoint(
  const std::string& arm, const std::vector<double>& joint_targets, const std::string& planner_profile,
  double velocity_scaling, double acceleration_scaling, ResultCallback callback)
{
  ExecuteSkill::Goal goal;
  goal.skill_name = "move_to_joint";
  goal.arm = arm;
  goal.planner_profile = planner_profile;
  goal.planning_mode = "normal";
  goal.joint_targets = joint_targets;
  goal.velocity_override = velocity_scaling;
  goal.acceleration_override = acceleration_scaling;
  goal.position_only = false;
  sendGoal(std::move(goal), std::move(callback));
}

void SkillClient::moveToJointSequence(
  const std::string& arm, const std::vector<double>& joint_sequence, const std::string& planner_profile,
  double velocity_scaling, double acceleration_scaling, ResultCallback callback)
{
  ExecuteSkill::Goal goal;
  goal.skill_name = "move_to_joint_sequence";
  goal.arm = arm;
  goal.planner_profile = planner_profile;
  goal.planning_mode = "normal";
  goal.joint_sequence = joint_sequence;
  goal.velocity_override = velocity_scaling;
  goal.acceleration_override = acceleration_scaling;
  goal.position_only = false;
  sendGoal(std::move(goal), std::move(callback));
}

void SkillClient::sendGoal(ExecuteSkill::Goal goal, ResultCallback callback)
{
  // robot_skills_node hosts this server behind a MoveItCpp stack (planning
  // scene, occupancy map monitor, etc.) that can take well over 5s to
  // finish initializing on a loaded machine - the old BT.CPP system never
  // hit this because its 50Hz tick loop kept retrying the connection
  // indefinitely across many ticks; this single-shot event-driven client
  // needs an explicitly generous wait instead of BT.CPP's incidental retry.
  if (!client_->wait_for_action_server(std::chrono::seconds(30))) {
    RCLCPP_ERROR(logger_, "ExecuteSkill action server not available after 30s");
    callback(false, "ExecuteSkill action server not available after 30s");
    return;
  }

  RCLCPP_INFO(
    logger_, "ExecuteSkill: %s | arm: %s | profile: %s", goal.skill_name.c_str(), goal.arm.c_str(),
    goal.planner_profile.c_str());

  rclcpp_action::Client<ExecuteSkill>::SendGoalOptions options;
  options.result_callback = [this, callback](const rclcpp_action::ClientGoalHandle<ExecuteSkill>::WrappedResult& result) {
    if (result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result->success) {
      callback(true, "");
      return;
    }
    const std::string msg = result.result ? result.result->error_message : "action did not succeed";
    RCLCPP_WARN(logger_, "ExecuteSkill failed: %s", msg.c_str());
    callback(false, msg);
  };
  options.goal_response_callback =
    [this, callback](const rclcpp_action::ClientGoalHandle<ExecuteSkill>::SharedPtr& handle) {
      if (!handle) {
        RCLCPP_ERROR(logger_, "ExecuteSkill goal rejected");
        callback(false, "goal rejected");
      }
    };

  client_->async_send_goal(goal, options);
}

}  // namespace sequence_executor
