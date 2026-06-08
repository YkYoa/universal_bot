#include "motion_planner/moveit_cpp_planner_manager.hpp"
#include <moveit/robot_state/conversions.hpp>
#include <moveit/robot_state/cartesian_interpolator.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>
#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <iomanip>
#include <sstream>

namespace {

const double TOLERANCE = 0.001;

const std::string PLAN_DIR = "/home/hans/universal_bot/plan";

std::string get_plan_filename(const planning_interface::PlannerRequest& request)
{
    std::stringstream ss;
    ss << request.getGroupName();
    if (request.getTargetPose().has_value()) {
        const auto& pose = request.getTargetPose().value().pose;
        ss << "_pose_" 
           << std::fixed << std::setprecision(4)
           << pose.position.x << "_" << pose.position.y << "_" << pose.position.z << "_"
           << pose.orientation.x << "_" << pose.orientation.y << "_" << pose.orientation.z << "_" << pose.orientation.w;
    } else if (!request.getJointTargets().empty()) {
        ss << "_joints";
        for (double val : request.getJointTargets()) {
            ss << "_" << std::fixed << std::setprecision(4) << val;
        }
    } else if (!request.getWaypoints().empty()) {
        ss << "_waypoints_" << request.getWaypoints().size();
        const auto& first = request.getWaypoints().front().pose;
        const auto& last = request.getWaypoints().back().pose;
        ss << "_from_" << std::fixed << std::setprecision(4) << first.position.x << "_" << first.position.y << "_" << first.position.z
           << "_to_" << last.position.x << "_" << last.position.y << "_" << last.position.z;
    }
    std::string s = ss.str();
    for (char& c : s) {
        if (c == ' ' || c == ',' || c == '/' || c == '\\' || c == ':') {
            c = '_';
        }
    }
    return s + ".yaml";
}

YAML::Node serialize_trajectory(const moveit_msgs::msg::RobotTrajectory& trajectory)
{
    YAML::Node node;
    for (const auto& name : trajectory.joint_trajectory.joint_names) {
        node["joint_names"].push_back(name);
    }
    for (const auto& pt : trajectory.joint_trajectory.points) {
        YAML::Node pt_node;
        for (double p : pt.positions) pt_node["positions"].push_back(p);
        for (double v : pt.velocities) pt_node["velocities"].push_back(v);
        for (double a : pt.accelerations) pt_node["accelerations"].push_back(a);
        for (double e : pt.effort) pt_node["effort"].push_back(e);
        pt_node["time_from_start"]["sec"] = pt.time_from_start.sec;
        pt_node["time_from_start"]["nanosec"] = pt.time_from_start.nanosec;
        node["points"].push_back(pt_node);
    }
    return node;
}

moveit_msgs::msg::RobotTrajectory deserialize_trajectory(const YAML::Node& node)
{
    moveit_msgs::msg::RobotTrajectory trajectory;
    if (node["joint_names"] && node["joint_names"].IsSequence()) {
        for (size_t i = 0; i < node["joint_names"].size(); ++i) {
            trajectory.joint_trajectory.joint_names.push_back(node["joint_names"][i].as<std::string>());
        }
    }
    if (node["points"] && node["points"].IsSequence()) {
        for (size_t i = 0; i < node["points"].size(); ++i) {
            YAML::Node pt_node = node["points"][i];
            trajectory_msgs::msg::JointTrajectoryPoint pt;
            if (pt_node["positions"]) {
                for (size_t j = 0; j < pt_node["positions"].size(); ++j) {
                    pt.positions.push_back(pt_node["positions"][j].as<double>());
                }
            }
            if (pt_node["velocities"]) {
                for (size_t j = 0; j < pt_node["velocities"].size(); ++j) {
                    pt.velocities.push_back(pt_node["velocities"][j].as<double>());
                }
            }
            if (pt_node["accelerations"]) {
                for (size_t j = 0; j < pt_node["accelerations"].size(); ++j) {
                    pt.accelerations.push_back(pt_node["accelerations"][j].as<double>());
                }
            }
            if (pt_node["effort"]) {
                for (size_t j = 0; j < pt_node["effort"].size(); ++j) {
                    pt.effort.push_back(pt_node["effort"][j].as<double>());
                }
            }
            if (pt_node["time_from_start"]) {
                pt.time_from_start.sec = pt_node["time_from_start"]["sec"].as<int32_t>();
                pt.time_from_start.nanosec = pt_node["time_from_start"]["nanosec"].as<uint32_t>();
            }
            trajectory.joint_trajectory.points.push_back(pt);
        }
    }
    return trajectory;
}

} // namespace

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

    std::string filename = get_plan_filename(request);
    std::filesystem::path plan_filepath = std::filesystem::path(PLAN_DIR) / filename;

    // Check if the plan is already cached
    if (std::filesystem::exists(plan_filepath)) {
        try {
            YAML::Node plan_node = YAML::LoadFile(plan_filepath.string());
            moveit_msgs::msg::RobotTrajectory traj = deserialize_trajectory(plan_node);
            bool start_state_matches = true;
            if (!traj.joint_trajectory.points.empty()) {
                const auto& traj_start_pts = traj.joint_trajectory.points[0].positions;
                const auto& traj_joint_names = traj.joint_trajectory.joint_names;
                if (traj_start_pts.size() == traj_joint_names.size()) {
                    moveit::core::RobotStatePtr start_state;
                    if (request.getStartState()) {
                        start_state = std::make_shared<moveit::core::RobotState>(*request.getStartState());
                    } else {
                        start_state = moveit_cpp_->getCurrentState();
                    }
                    if (start_state) {
                        for (size_t i = 0; i < traj_joint_names.size(); ++i) {
                            if (start_state->getRobotModel()->hasJointModel(traj_joint_names[i])) {
                                double traj_val = traj_start_pts[i];
                                double current_val = start_state->getVariablePosition(traj_joint_names[i]);
                                if (std::abs(traj_val - current_val) > TOLERANCE) { // 0.009 rad tolerance (slightly tighter than MoveIt's 0.01)
                                    RCLCPP_WARN(node_->get_logger(),
                                        "[MoveItCppPlannerManager] Start state discrepancy for joint %s: traj_val=%.4f, current_val=%.4f. Bypassing cached plan.",
                                        traj_joint_names[i].c_str(), traj_val, current_val);
                                    start_state_matches = false;
                                    break;
                                }
                            } else {
                                start_state_matches = false;
                                break;
                            }
                        }
                    } else {
                        start_state_matches = false;
                    }
                } else {
                    start_state_matches = false;
                }
            } else {
                start_state_matches = false;
            }

            if (start_state_matches) {
                response.trajectory = traj;
                response.success = true;
                response.planning_time = 0.001;
                RCLCPP_INFO(node_->get_logger(),
                    "[MoveItCppPlannerManager] Loaded saved plan from: %s", 
                    plan_filepath.string().c_str());
                return response;
            } else {
                RCLCPP_INFO(node_->get_logger(),
                    "[MoveItCppPlannerManager] Cached plan in %s start state does not match. Will replan.",
                    plan_filepath.string().c_str());
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(node_->get_logger(),
                "[MoveItCppPlannerManager] Failed to load saved plan '%s': %s. Will replan.",
                plan_filepath.string().c_str(), e.what());
        }
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

    // RCLCPP_INFO(node_->get_logger(),
    //     "[MoveItCppPlannerManager] Planning request details:\n"
    //     "  - Group: %s\n"
    //     "  - Profile Name: %s\n"
    //     "  - Profile Pipeline ID: %s\n"
    //     "  - Profile Planner ID: %s\n"
    //     "  - Parameter override Pipeline ID: %s\n"
    //     "  - Parameter override Planner ID: %s\n"
    //     "  - Selected Pipeline ID: %s\n"
    //     "  - Selected Planner ID: %s",
    //     request.getGroupName().c_str(),
    //     profile_name.c_str(),
    //     profile.pipeline_id.c_str(),
    //     profile.planner_id.c_str(),
    //     overrides.pipeline_id.c_str(),
    //     overrides.planner_id.c_str(),
    //     plan_params.planning_pipeline.c_str(),
    //     plan_params.planner_id.c_str());

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

        if (request.isPositionOnly() && plan_params.planning_pipeline != "pilz_industrial_motion_planner") {
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
        std::string ee_link = ee_link_for_group(request.getGroupName());
        if (ee_link.empty()) {
            response.success = false;
            response.error_message = "Could not resolve end-effector link for group " + request.getGroupName();
            return response;
        }

        auto jmg = moveit_cpp_->getRobotModel()->getJointModelGroup(request.getGroupName());
        if (!jmg) {
            response.success = false;
            response.error_message = "Could not find joint model group " + request.getGroupName();
            return response;
        }

        auto link_model = moveit_cpp_->getRobotModel()->getLinkModel(ee_link);
        if (!link_model) {
            response.success = false;
            response.error_message = "Could not find link model " + ee_link;
            return response;
        }

        moveit::core::RobotStatePtr start_state;
        if (request.getStartState()) {
            start_state = std::make_shared<moveit::core::RobotState>(*request.getStartState());
        } else {
            start_state = moveit_cpp_->getCurrentState();
        }

        if (!start_state) {
            response.success = false;
            response.error_message = "Failed to acquire starting RobotState.";
            return response;
        }

        EigenSTL::vector_Isometry3d eigen_waypoints;
        eigen_waypoints.reserve(request.getWaypoints().size());
        for (const auto& ps : request.getWaypoints()) {
            Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
            t.translation() = Eigen::Vector3d(ps.pose.position.x, ps.pose.position.y, ps.pose.position.z);
            t.linear() = Eigen::Quaterniond(ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z).toRotationMatrix();
            eigen_waypoints.push_back(t);
        }

        std::vector<moveit::core::RobotStatePtr> traj_states;
        moveit::core::MaxEEFStep max_step(0.01);
        moveit::core::CartesianPrecision precision;

        double fraction = moveit::core::CartesianInterpolator::computeCartesianPath(
            start_state.get(),
            jmg,
            traj_states,
            link_model,
            eigen_waypoints,
            true,
            max_step,
            precision
        );

        if (fraction <= 0.0 || traj_states.empty()) {
            response.success = false;
            response.error_message = "Cartesian planning failed completely (fraction: " + std::to_string(fraction) + ")";
            return response;
        }

        auto rt = std::make_shared<robot_trajectory::RobotTrajectory>(moveit_cpp_->getRobotModel(), request.getGroupName());
        for (const auto& state : traj_states) {
            rt->addSuffixWayPoint(state, 0.0);
        }

        trajectory_processing::TimeOptimalTrajectoryGeneration totg;
        if (!totg.computeTimeStamps(*rt, profile.velocity_scaling, profile.acceleration_scaling)) {
            response.success = false;
            response.error_message = "Time parameterization failed for Cartesian path.";
            return response;
        }

        response.success = true;
        rt->getRobotTrajectoryMsg(response.trajectory);
        response.planning_time = 0.01;

        // Save successfully run Cartesian plan
        try {
            std::filesystem::path plan_dir(PLAN_DIR);
            if (!std::filesystem::exists(plan_dir)) {
                std::filesystem::create_directories(plan_dir);
            }
            YAML::Node plan_node = serialize_trajectory(response.trajectory);
            std::ofstream fout(plan_filepath.string());
            fout << plan_node;
            fout.close();
            RCLCPP_INFO(node_->get_logger(),
                "[MoveItCppPlannerManager] Saved successfully run plan to: %s",
                plan_filepath.string().c_str());
        } catch (const std::exception& e) {
            RCLCPP_WARN(node_->get_logger(),
                "[MoveItCppPlannerManager] Failed to save plan to '%s': %s",
                plan_filepath.string().c_str(), e.what());
        }

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

        // Save successfully run standard plan
        try {
            std::filesystem::path plan_dir(PLAN_DIR);
            if (!std::filesystem::exists(plan_dir)) {
                std::filesystem::create_directories(plan_dir);
            }
            YAML::Node plan_node = serialize_trajectory(response.trajectory);
            std::ofstream fout(plan_filepath.string());
            fout << plan_node;
            fout.close();
            RCLCPP_INFO(node_->get_logger(),
                "[MoveItCppPlannerManager] Saved successfully run plan to: %s",
                plan_filepath.string().c_str());
        } catch (const std::exception& e) {
            RCLCPP_WARN(node_->get_logger(),
                "[MoveItCppPlannerManager] Failed to save plan to '%s': %s",
                plan_filepath.string().c_str(), e.what());
        }
    } else {
        response.success = false;
        response.error_message = "Planning failed for group " + request.getGroupName();
    }

    return response;
}

} // namespace motion_planner
