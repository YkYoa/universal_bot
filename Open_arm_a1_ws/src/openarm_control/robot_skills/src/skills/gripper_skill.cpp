#include "robot_skills/skills/gripper_skill.hpp"

namespace robot_skills
{

bool GripperSkill::initialize(
    const std::shared_ptr<rclcpp::Node>& node,
    const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner)
{
    node_ = node;
    planner_ = planner;
    return true;
}

SkillResult GripperSkill::execute(
    const SkillRequest& req,
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<openarm_messages::action::ExecuteSkill>>& goal_handle)
{
    SkillResult result;

    if (!planner_ || !planner_->getMoveItCpp()) {
        result.success = false;
        result.error_message = "Planner/MoveItCpp is not initialized.";
        return result;
    }

    // Determine target gripper planning group based on request arm
    std::string gripper_group = (req.arm == "left_arm") ? "left_gripper" : "right_gripper";
    std::string target_state = (skill_name_ == "open_gripper") ? "open" : "close";

    RCLCPP_INFO(node_->get_logger(), "[GripperSkill] Planning gripper %s to state '%s'", 
                gripper_group.c_str(), target_state.c_str());

    // 1. Get default positions for the state from SRDF
    auto robot_model = planner_->getMoveItCpp()->getRobotModel();
    if (!robot_model) {
        result.success = false;
        result.error_message = "Robot model is null.";
        return result;
    }

    auto joint_model_group = robot_model->getJointModelGroup(gripper_group);
    if (!joint_model_group) {
        result.success = false;
        result.error_message = "JointModelGroup '" + gripper_group + "' not found.";
        return result;
    }

    std::map<std::string, double> joint_map;
    if (!joint_model_group->getVariableDefaultPositions(target_state, joint_map)) {
        result.success = false;
        result.error_message = "Named pose '" + target_state + "' not found in group '" + gripper_group + "'";
        RCLCPP_ERROR(node_->get_logger(), "[GripperSkill] %s", result.error_message.c_str());
        return result;
    }
    std::vector<double> joint_values;
    joint_values.reserve(joint_map.size());
    for (const auto& name : joint_model_group->getVariableNames()) {
        joint_values.push_back(joint_map[name]);
    }

    // 2. Build planning request
    planning_interface::PlannerRequest plan_req;
    plan_req.setGroupName(gripper_group);
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
        RCLCPP_ERROR(node_->get_logger(), "[GripperSkill] Gripper planning failed: %s", result.error_message.c_str());
        return result;
    }

    RCLCPP_INFO(node_->get_logger(), "[GripperSkill] Gripper planning succeeded. Executing...");

    // 4. Dispatch execution to server helper
    return server_->execute_trajectory(gripper_group, plan_resp.trajectory, req.mode, goal_handle, req);
}

} // namespace robot_skills
