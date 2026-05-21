#include "motion_planner/moveit_cpp_planner_manager.hpp"
#include <moveit/robot_state/conversions.hpp>

namespace motion_planner
{

bool MoveItCppPlannerManager::initialize(const std::shared_ptr<rclcpp::Node>& node)
{
    node_ = node;

    // Initialize MoveItCpp options
    moveit_cpp::MoveItCpp::Options opts(node);
    opts.planning_scene_monitor_options.name = "planning_scene_monitor";
    opts.planning_scene_monitor_options.robot_description = "robot_description";

    RCLCPP_INFO(node_->get_logger(), "[MoveItCppPlannerManager] Initializing MoveItCpp...");
    moveit_cpp_ = std::make_shared<moveit_cpp::MoveItCpp>(node_, opts);
    moveit_cpp_->getPlanningSceneMonitorNonConst()->providePlanningSceneService();

    RCLCPP_INFO(node_->get_logger(), "[MoveItCppPlannerManager] Loading planner profiles...");
    profile_loader_ = planning_interface::PlannerProfileLoader::from_package("motion_planner");

    RCLCPP_INFO(node_->get_logger(), "[MoveItCppPlannerManager] Initialization complete.");
    return true;
}

std::string MoveItCppPlannerManager::ee_link_for_group(const std::string& group) const
{
    if (group == "left_arm") {
        return "openarm_left_hand_tcp";
    } else if (group == "right_arm") {
        return "openarm_right_hand_tcp";
    }

    // Dynamic fallback using JointModelGroup end effector tips
    if (moveit_cpp_ && moveit_cpp_->getRobotModel()) {
        auto jmg = moveit_cpp_->getRobotModel()->getJointModelGroup(group);
        if (jmg) {
            std::vector<std::string> tips;
            jmg->getEndEffectorTips(tips);
            if (!tips.empty()) {
                return tips.front();
            }
        }
    }
    return "";
}

planning_interface::PlannerResponse MoveItCppPlannerManager::plan(const planning_interface::PlannerRequest& request)
{
    planning_interface::PlannerResponse response;

    if (!moveit_cpp_) {
        response.success = false;
        response.error_message = "MoveItCpp is not initialized.";
        return response;
    }

    // 1. Load the planner profile
    std::string profile_name = request.getProfileName();
    planning_interface::PlannerProfile profile;
    auto opt_profile = profile_loader_->get_profile(profile_name);
    if (opt_profile) {
        profile = *opt_profile;
    } else {
        RCLCPP_WARN(node_->get_logger(), 
            "[MoveItCppPlannerManager] Profile '%s' not found. Using default profile.", 
            profile_name.c_str());
        profile = profile_loader_->get_default_profile();
    }

    // 2. Apply request parameter overrides if specified
    const auto& overrides = request.getParameters();
    if (!overrides.pipeline_id.empty()) {
        profile.pipeline_id = overrides.pipeline_id;
    }
    if (!overrides.planner_id.empty()) {
        profile.planner_id = overrides.planner_id;
    }
    if (overrides.planning_attempts > 0) {
        profile.num_attempts = overrides.planning_attempts;
    }
    if (overrides.planning_time > 0.0) {
        profile.planning_time = overrides.planning_time;
    }
    if (overrides.velocity_scaling > 0.0) {
        profile.velocity_scaling = overrides.velocity_scaling;
    }
    if (overrides.acceleration_scaling > 0.0) {
        profile.acceleration_scaling = overrides.acceleration_scaling;
    }

    // 3. Create planning component
    auto planning_component = std::make_shared<moveit_cpp::PlanningComponent>(
        request.getGroupName(), moveit_cpp_);

    // 4. Configure plan request parameters
    moveit_cpp::PlanningComponent::PlanRequestParameters plan_params;
    plan_params.planning_pipeline = profile.pipeline_id;
    plan_params.planner_id = profile.planner_id;
    plan_params.planning_attempts = profile.num_attempts;
    plan_params.planning_time = profile.planning_time;
    plan_params.max_velocity_scaling_factor = profile.velocity_scaling;
    plan_params.max_acceleration_scaling_factor = profile.acceleration_scaling;

    // 5. Configure start state
    if (request.getStartState()) {
        planning_component->setStartState(*request.getStartState());
    } else {
        planning_component->setStartStateToCurrentState();
    }

    // 6. Set goals
    if (request.getTargetPose().has_value()) {
        std::string ee_link = ee_link_for_group(request.getGroupName());
        if (ee_link.empty()) {
            response.success = false;
            response.error_message = "Could not resolve end-effector link for group " + request.getGroupName();
            return response;
        }

        if (request.isPositionOnly()) {
            // Apply position-only constraint using moveit Constraints manually if needed, 
            // or we can set it and rely on position constraints.
            // MoveItCpp PlanningComponent lets us set goal with constraint region.
            moveit_msgs::msg::Constraints constraints;
            moveit_msgs::msg::PositionConstraint pc;
            pc.header.frame_id = request.getTargetPose()->header.frame_id;
            pc.link_name = ee_link;
            pc.weight = 1.0;

            shape_msgs::msg::SolidPrimitive sphere;
            sphere.type = shape_msgs::msg::SolidPrimitive::SPHERE;
            sphere.dimensions = { 0.01 }; // 1 cm tolerance

            moveit_msgs::msg::BoundingVolume bv;
            bv.primitives.push_back(sphere);
            bv.primitive_poses.push_back(request.getTargetPose()->pose);
            pc.constraint_region = bv;
            constraints.position_constraints.push_back(pc);

            std::vector<moveit_msgs::msg::Constraints> goal_constraints = { constraints };
            planning_component->setGoal(goal_constraints);
        } else {
            planning_component->setGoal(request.getTargetPose().value(), ee_link);
        }
    } else if (!request.getJointTargets().empty()) {
        moveit::core::RobotState goal_state(moveit_cpp_->getRobotModel());
        goal_state.setToDefaultValues();
        goal_state.setJointGroupPositions(request.getGroupName(), request.getJointTargets());
        goal_state.update();
        planning_component->setGoal(goal_state);
    } else if (!request.getWaypoints().empty()) {
        // Cartesian path planning is not yet implemented via PlanningComponent.
        // computeCartesianPath must be done via moveit::core::RobotState directly.
        // Returning failure here is intentional — do NOT fall through with no goal set,
        // as that would cause MoveIt to plan vacuously or crash.
        RCLCPP_ERROR(node_->get_logger(),
            "[MoveItCppPlannerManager] Waypoint/Cartesian planning is not yet implemented. "
            "Use CartesianMoveSkill with a single target_pose instead, or implement "
            "computeCartesianPath via moveit::core::RobotState.");
        response.success = false;
        response.error_message = "Waypoint/Cartesian planning not yet implemented.";
        return response;
    } else {
        response.success = false;
        response.error_message = "No valid goal (pose or joints) specified in request.";
        return response;
    }

    // 7. Perform planning
    auto plan_solution = planning_component->plan(plan_params);
    if (plan_solution) {
        response.success = true;
        plan_solution.trajectory->getRobotTrajectoryMsg(response.trajectory);
        response.planning_time = plan_solution.planning_time;
    } else {
        response.success = false;
        response.error_message = "Planning failed for group " + request.getGroupName();
    }

    return response;
}

} // namespace motion_planner
