#include "robot_skills/skills/move_to_joint_skill.hpp"
#include <moveit/robot_state/robot_state.hpp>

namespace robot_skills
{

bool MoveToJointSkill::initialize(
    const std::shared_ptr<rclcpp::Node>& node,
    const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner)
{
    node_ = node;
    planner_ = planner;
    return true;
}

SkillResult MoveToJointSkill::execute(
    const SkillRequest& req,
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<openarm_messages::action::ExecuteSkill>>& goal_handle)
{
    SkillResult result;

    if (!planner_ || !planner_->getMoveItCpp()) {
        result.success = false;
        result.error_message = "Planner/MoveItCpp is not initialized.";
        return result;
    }

    RCLCPP_INFO(node_->get_logger(), "[MoveToJointSkill] Planning to joint targets for arm: %s", req.arm.c_str());

    if (req.joint_target.name.empty()) {
        result.success = false;
        result.error_message = "Joint targets are empty.";
        RCLCPP_ERROR(node_->get_logger(), "[MoveToJointSkill] %s", result.error_message.c_str());
        return result;
    }
    if (req.joint_target.name.size() != req.joint_target.position.size()) {
        result.success = false;
        result.error_message = "joint_target name/position length mismatch (" +
            std::to_string(req.joint_target.name.size()) + " names, " +
            std::to_string(req.joint_target.position.size()) + " positions).";
        RCLCPP_ERROR(node_->get_logger(), "[MoveToJointSkill] %s", result.error_message.c_str());
        return result;
    }

    auto robot_model = planner_->getMoveItCpp()->getRobotModel();
    const auto* jmg = robot_model->getJointModelGroup(req.arm);
    if (!jmg) {
        result.success = false;
        result.error_message = "Unknown planning group: " + req.arm;
        RCLCPP_ERROR(node_->get_logger(), "[MoveToJointSkill] %s", result.error_message.c_str());
        return result;
    }

    // Resolve name->position against the live robot model, same pattern as
    // moveToNamedPose (see skill_client.hpp's doc comment): a name the live
    // model doesn't have (e.g. amazing_hand's finger_joint1 under
    // ee_type:=none) is skipped rather than rejected, so a joint_target
    // authored for the "other" ee_type still works for whichever joints DO
    // exist - no positional stride/DOF assumption anywhere in this path.
    moveit::core::RobotState goal_state(robot_model);
    goal_state.setToDefaultValues();
    std::size_t matched = 0;
    for (std::size_t i = 0; i < req.joint_target.name.size(); ++i) {
        if (robot_model->hasJointModel(req.joint_target.name[i])) {
            goal_state.setVariablePosition(req.joint_target.name[i], req.joint_target.position[i]);
            ++matched;
        }
    }
    if (matched == 0) {
        result.success = false;
        result.error_message = "None of the joint_target names exist on the live robot model.";
        RCLCPP_ERROR(node_->get_logger(), "[MoveToJointSkill] %s", result.error_message.c_str());
        return result;
    }
    goal_state.update();

    std::vector<double> group_positions;
    goal_state.copyJointGroupPositions(jmg, group_positions);

    // 1. Build planning request
    planning_interface::PlannerRequest plan_req;
    plan_req.setGroupName(req.arm);
    plan_req.setProfileName(req.planner_profile);
    plan_req.setJointTargets(group_positions);

    // Apply scaling overrides
    planning_interface::PlanRequestParameters params;
    if (req.velocity_override > 0.0 || req.acceleration_override > 0.0) {
        params.velocity_scaling = req.velocity_override;
        params.acceleration_scaling = req.acceleration_override;
        plan_req.setParameters(params);
    }

    // 2. Perform planning
    auto plan_resp = planner_->plan(plan_req);
    result.planning_time_sec = plan_resp.planning_time;

    if (!plan_resp.success) {
        result.success = false;
        result.error_message = plan_resp.error_message;
        RCLCPP_ERROR(node_->get_logger(), "[MoveToJointSkill] Planning failed: %s", result.error_message.c_str());
        return result;
    }

    RCLCPP_INFO(node_->get_logger(), "[MoveToJointSkill] Planning succeeded. Executing trajectory...");

    // 3. Dispatch execution to server helper
    return server_->execute_trajectory(req.arm, plan_resp.trajectory, req.mode, goal_handle, req);
}

} // namespace robot_skills
