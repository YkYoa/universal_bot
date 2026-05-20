#include "robot_skills/skills/move_to_named_pose_skill.hpp"

namespace robot_skills
{

bool MoveToNamedPoseSkill::initialize(
    const std::shared_ptr<rclcpp::Node>& node,
    const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner)
{
    node_ = node;
    planner_ = planner;
    return true;
}

SkillResult MoveToNamedPoseSkill::execute(
    const SkillRequest& req,
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<openarm_messages::action::ExecuteSkill>>& goal_handle)
{
    SkillResult result;

    if (!planner_ || !planner_->getMoveItCpp()) {
        result.success = false;
        result.error_message = "Planner/MoveItCpp is not initialized.";
        return result;
    }

    RCLCPP_INFO(node_->get_logger(), "[MoveToNamedPoseSkill] Planning to named pose '%s' for arm: %s", 
                req.named_pose.c_str(), req.arm.c_str());

    // 1. Get joint values for the named pose from the SRDF dynamically
    auto robot_model = planner_->getMoveItCpp()->getRobotModel();
    if (!robot_model) {
        result.success = false;
        result.error_message = "Robot model is null.";
        return result;
    }

    auto joint_model_group = robot_model->getJointModelGroup(req.arm);
    if (!joint_model_group) {
        result.success = false;
        result.error_message = "JointModelGroup '" + req.arm + "' not found.";
        return result;
    }

    std::map<std::string, double> joint_map;
    if (!joint_model_group->getVariableDefaultPositions(req.named_pose, joint_map)) {
        result.success = false;
        result.error_message = "Named pose '" + req.named_pose + "' not found in group '" + req.arm + "'";
        RCLCPP_ERROR(node_->get_logger(), "[MoveToNamedPoseSkill] %s", result.error_message.c_str());
        return result;
    }
    std::vector<double> joint_values;
    joint_values.reserve(joint_map.size());
    for (const auto& name : joint_model_group->getVariableNames()) {
        joint_values.push_back(joint_map[name]);
    }

    // 2. Build planning request
    planning_interface::PlannerRequest plan_req;
    plan_req.setGroupName(req.arm);
    plan_req.setProfileName(req.planner_profile);
    plan_req.setJointTargets(joint_values);

    // Apply scaling overrides
    planning_interface::PlanRequestParameters params;
    if (req.velocity_override > 0.0) {
        params.velocity_scaling = req.velocity_override;
        plan_req.setParameters(params);
    }

    // 3. Perform planning
    auto plan_resp = planner_->plan(plan_req);
    result.planning_time_sec = plan_resp.planning_time;

    if (!plan_resp.success) {
        result.success = false;
        result.error_message = plan_resp.error_message;
        RCLCPP_ERROR(node_->get_logger(), "[MoveToNamedPoseSkill] Planning failed: %s", result.error_message.c_str());
        return result;
    }

    RCLCPP_INFO(node_->get_logger(), "[MoveToNamedPoseSkill] Planning succeeded. Executing trajectory...");

    // 4. Dispatch execution to server helper
    return server_->execute_trajectory(req.arm, plan_resp.trajectory, req.mode, goal_handle, req);
}

} // namespace robot_skills
