#include "motion_planner/moveit_cpp_planner_manager.hpp"
#include <algorithm>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/robot_state/cartesian_interpolator.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>
#include <moveit/trajectory_processing/ruckig_traj_smoothing.hpp>
#include <moveit/planning_scene_monitor/planning_scene_monitor.hpp>
#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <iomanip>
#include <sstream>
#include <Eigen/Geometry>

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
    } else if (!request.getJointSequence().empty()) {
        ss << "_jointseq_" << request.getJointSequence().size();
        ss << "_from";
        for (double v : request.getJointSequence().front()) {
            ss << "_" << std::fixed << std::setprecision(4) << v;
        }
        ss << "_to";
        for (double v : request.getJointSequence().back()) {
            ss << "_" << std::fixed << std::setprecision(4) << v;
        }
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

// Rounds the sharp direction-change at each interior waypoint of a joint-
// space path with a local quadratic Bezier corner-cut, instead of routing
// exactly through it. For each corner index c (not the first/last state in
// `states`), takes a point a few samples before it (P_before) and a few
// samples after it (P_after) - both already inside the neighboring,
// already-planned segments - and replaces the points between them with
// samples along B(t) = (1-t)^2*P_before + 2t(1-t)*P_apex + t^2*P_after per
// joint. This starts exactly at P_before, ends exactly at P_after (so it
// splices cleanly into the untouched rest of the path), and passes near -
// not exactly over - the original corner, eliminating the vertex that
// forces TOTG/Ruckig to slow down there. Blend windows are clamped so they
// never overlap a neighboring corner or run off the ends of the path; a
// corner whose neighbors are too close to leave any room is left unblended.
std::vector<moveit::core::RobotStatePtr> blendJointSequenceCorners(
    const std::vector<moveit::core::RobotStatePtr>& states, const std::vector<size_t>& corner_indices,
    const std::string& group_name)
{
    constexpr size_t kBlendWindow = 5;  // points on each side of a corner, before clamping

    std::vector<moveit::core::RobotStatePtr> blended = states;

    for (size_t c = 0; c < corner_indices.size(); ++c) {
        const size_t corner_idx = corner_indices[c];
        const size_t prev_bound = (c == 0) ? 0 : corner_indices[c - 1];
        const size_t next_bound = (c + 1 < corner_indices.size()) ? corner_indices[c + 1] : (states.size() - 1);

        const size_t window_before = std::min(kBlendWindow, (corner_idx - prev_bound) / 2);
        const size_t window_after = std::min(kBlendWindow, (next_bound - corner_idx) / 2);
        if (window_before == 0 || window_after == 0) {
            continue;  // neighboring corners too close together to blend safely
        }

        const size_t before_idx = corner_idx - window_before;
        const size_t after_idx = corner_idx + window_after;

        std::vector<double> p_before, p_apex, p_after;
        states[before_idx]->copyJointGroupPositions(group_name, p_before);
        states[corner_idx]->copyJointGroupPositions(group_name, p_apex);
        states[after_idx]->copyJointGroupPositions(group_name, p_after);

        const size_t num_points = after_idx - before_idx + 1;
        for (size_t k = 0; k < num_points; ++k) {
            const double t = static_cast<double>(k) / static_cast<double>(num_points - 1);
            const double one_minus_t = 1.0 - t;

            std::vector<double> blended_vals(p_apex.size());
            for (size_t d = 0; d < p_apex.size(); ++d) {
                blended_vals[d] = one_minus_t * one_minus_t * p_before[d] + 2.0 * t * one_minus_t * p_apex[d] +
                    t * t * p_after[d];
            }

            auto blended_state = std::make_shared<moveit::core::RobotState>(*states[before_idx]);
            blended_state->setJointGroupPositions(group_name, blended_vals);
            blended_state->update();
            blended[before_idx + k] = blended_state;
        }
    }

    return blended;
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
    auto psm = moveit_cpp_->getPlanningSceneMonitorNonConst();
    if (psm) {
        psm->startSceneMonitor();
        psm->startWorldGeometryMonitor();
        psm->providePlanningSceneService();
    }

    RCLCPP_INFO(node_->get_logger(), "[MoveItCppPlannerManager] Loading planner profiles...");
    profile_loader_ = planning_interface::PlannerProfileLoader::from_package("motion_planner");

    target_marker_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>("/motion_planner/target_pose_marker", 10);
    trajectory_marker_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>("/motion_planner/planned_trajectory_markers", 10);
    display_pub_ = node_->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 10);

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
        publish_target_marker_for_request(request, false);
        publish_trajectory_markers(moveit_msgs::msg::RobotTrajectory(), request.getGroupName(), false);
        return response;
    }

    bool save_plan = true;
    std::string resolved_profile_name = request.getProfileName();
    size_t pos = resolved_profile_name.find("_no_save");
    if (pos != std::string::npos) {
        save_plan = false;
        resolved_profile_name = resolved_profile_name.substr(0, pos);
    }

    std::string filename = get_plan_filename(request);
    std::filesystem::path plan_filepath = std::filesystem::path(PLAN_DIR) / filename;

    // Check if the plan is already cached
    if (save_plan && std::filesystem::exists(plan_filepath)) {
        try {
            YAML::Node plan_node = YAML::LoadFile(plan_filepath.string());
            moveit_msgs::msg::RobotTrajectory traj = deserialize_trajectory(plan_node);
            bool start_state_matches = true;
            moveit::core::RobotStatePtr start_state;
            if (request.getStartState()) {
                start_state = std::make_shared<moveit::core::RobotState>(*request.getStartState());
            } else {
                start_state = moveit_cpp_->getCurrentState();
            }

            if (!traj.joint_trajectory.points.empty()) {
                const auto& traj_start_pts = traj.joint_trajectory.points[0].positions;
                const auto& traj_joint_names = traj.joint_trajectory.joint_names;
                if (traj_start_pts.size() == traj_joint_names.size()) {
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
                // Check if the cached trajectory is still valid (collision-free) in the current planning scene.
                bool trajectory_valid = true;
                if (moveit_cpp_->getPlanningSceneMonitorNonConst()) {
                    planning_scene_monitor::LockedPlanningSceneRO planning_scene(moveit_cpp_->getPlanningSceneMonitorNonConst());
                    if (planning_scene) {
                        robot_trajectory::RobotTrajectory robot_traj(moveit_cpp_->getRobotModel(), request.getGroupName());
                        robot_traj.setRobotTrajectoryMsg(*start_state, traj);
                        if (!planning_scene->isPathValid(robot_traj, request.getGroupName())) {
                            RCLCPP_WARN(node_->get_logger(),
                                "[MoveItCppPlannerManager] Cached plan in %s is in collision or invalid in current planning scene. Bypassing cached plan.",
                                plan_filepath.string().c_str());
                            trajectory_valid = false;
                        }
                    }
                }

                if (trajectory_valid) {
                    response.trajectory = traj;
                    response.success = true;
                    response.planning_time = 0.001;
                    RCLCPP_INFO(node_->get_logger(),
                        "[MoveItCppPlannerManager] Loaded saved plan from: %s", 
                        plan_filepath.string().c_str());
                    publish_target_marker_for_request(request, true);
                    publish_trajectory_markers(response.trajectory, request.getGroupName(), true);

                    // Publish display trajectory
                    moveit_msgs::msg::DisplayTrajectory display_msg;
                    display_msg.model_id = moveit_cpp_->getRobotModel()->getName();
                    display_msg.trajectory.push_back(response.trajectory);
                    if (start_state) {
                        moveit::core::robotStateToRobotStateMsg(*start_state, display_msg.trajectory_start);
                    }
                    if (display_pub_) {
                        display_pub_->publish(display_msg);
                    }

                    return response;
                } else {
                    start_state_matches = false;
                }
            }

            if (!start_state_matches) {
                RCLCPP_INFO(node_->get_logger(),
                    "[MoveItCppPlannerManager] Cached plan in %s start state does not match or is invalid. Will replan.",
                    plan_filepath.string().c_str());
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(node_->get_logger(),
                "[MoveItCppPlannerManager] Failed to load saved plan '%s': %s. Will replan.",
                plan_filepath.string().c_str(), e.what());
        }
    }

    // 1. Load the planner profile
    planning_interface::PlannerProfile profile;
    auto opt_profile = profile_loader_->get_profile(resolved_profile_name);
    if (opt_profile) {
        profile = *opt_profile;
    } else {
        RCLCPP_WARN(node_->get_logger(), 
            "[MoveItCppPlannerManager] Profile '%s' not found. Using default profile.", 
            resolved_profile_name.c_str());
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
            publish_target_marker_for_request(request, false);
            publish_trajectory_markers(moveit_msgs::msg::RobotTrajectory(), request.getGroupName(), false);
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
    } else if (!request.getJointSequence().empty()) {
        // Plan each waypoint segment individually (collision-aware, same as
        // a normal joint-target plan), but concatenate every segment's
        // states into ONE robot_trajectory::RobotTrajectory and re-time the
        // whole thing with a single TOTG pass at the end - this is exactly
        // the pattern already used below for Cartesian `waypoints`, applied
        // to joint-space. TOTG only forces zero velocity at the very first
        // and last point of the whole path, not at interior waypoints, so
        // this produces one continuous motion instead of N stop-start
        // segments.
        const auto& joint_sequence = request.getJointSequence();

        moveit::core::RobotStatePtr start_state;
        if (request.getStartState()) {
            start_state = std::make_shared<moveit::core::RobotState>(*request.getStartState());
        } else {
            start_state = moveit_cpp_->getCurrentState();
        }

        if (!start_state) {
            response.success = false;
            response.error_message = "Failed to acquire starting RobotState.";
            publish_target_marker_for_request(request, false);
            publish_trajectory_markers(moveit_msgs::msg::RobotTrajectory(), request.getGroupName(), false);
            return response;
        }

        std::vector<moveit::core::RobotStatePtr> traj_states;
        std::vector<size_t> corner_indices;  // traj_states index of each interior recorded waypoint
        moveit::core::RobotState active_state(*start_state);

        for (size_t i = 0; i < joint_sequence.size(); ++i) {
            planning_component->setStartState(active_state);

            moveit::core::RobotState goal_state(active_state);
            goal_state.setJointGroupPositions(request.getGroupName(), joint_sequence[i]);
            goal_state.update();
            planning_component->setGoal(goal_state);

            auto seg_solution = planning_component->plan(plan_params);
            if (!seg_solution) {
                RCLCPP_ERROR(node_->get_logger(),
                             "[MoveItCppPlannerManager] Joint-sequence planning failed at waypoint %zu of %zu",
                             i, joint_sequence.size());
                response.success = false;
                response.error_message = "Joint-sequence planning failed at waypoint " + std::to_string(i);
                publish_target_marker_for_request(request, false);
                publish_trajectory_markers(moveit_msgs::msg::RobotTrajectory(), request.getGroupName(), false);
                return response;
            }

            // Each segment's trajectory includes both endpoints - skip
            // index 0 for every segment after the first to avoid a
            // duplicate (zero-duration) waypoint at the join.
            size_t start_idx = (i == 0) ? 0 : 1;
            for (size_t j = start_idx; j < seg_solution.trajectory->getWayPointCount(); ++j) {
                traj_states.push_back(std::make_shared<moveit::core::RobotState>(seg_solution.trajectory->getWayPoint(j)));
            }
            active_state = *traj_states.back();
            if (i + 1 < joint_sequence.size()) {
                // This is an interior waypoint (not the very last target) -
                // a corner-blend candidate.
                corner_indices.push_back(traj_states.size() - 1);
            }
        }

        // Round the sharp direction-change at each interior waypoint (a
        // local quadratic Bezier corner-cut, see blendJointSequenceCorners)
        // instead of routing exactly through it - otherwise TOTG/Ruckig
        // correctly (but undesirably) slow the arm to near-zero velocity at
        // any waypoint where incoming/outgoing direction differs sharply.
        // Revalidate against the planning scene and fall back to the exact
        // (already collision-checked) path if the blend isn't safe.
        std::vector<moveit::core::RobotStatePtr> final_states = traj_states;
        if (!corner_indices.empty()) {
            auto blended_states = blendJointSequenceCorners(traj_states, corner_indices, request.getGroupName());

            auto blended_rt = std::make_shared<robot_trajectory::RobotTrajectory>(moveit_cpp_->getRobotModel(), request.getGroupName());
            for (const auto& state : blended_states) {
                blended_rt->addSuffixWayPoint(state, 0.0);
            }

            bool blend_valid = true;
            if (moveit_cpp_->getPlanningSceneMonitorNonConst()) {
                planning_scene_monitor::LockedPlanningSceneRO planning_scene(moveit_cpp_->getPlanningSceneMonitorNonConst());
                if (planning_scene && !planning_scene->isPathValid(*blended_rt, request.getGroupName())) {
                    blend_valid = false;
                }
            }

            if (blend_valid) {
                final_states = blended_states;
            } else {
                RCLCPP_WARN(node_->get_logger(),
                    "[MoveItCppPlannerManager] Corner-blended joint sequence path is in collision - "
                    "using the exact (unblended) path instead.");
            }
        }

        auto rt = std::make_shared<robot_trajectory::RobotTrajectory>(moveit_cpp_->getRobotModel(), request.getGroupName());
        for (const auto& state : final_states) {
            rt->addSuffixWayPoint(state, 0.0);
        }

        trajectory_processing::TimeOptimalTrajectoryGeneration totg;
        if (!totg.computeTimeStamps(*rt, profile.velocity_scaling, profile.acceleration_scaling)) {
            response.success = false;
            response.error_message = "Time parameterization failed for joint sequence.";
            publish_target_marker_for_request(request, false);
            publish_trajectory_markers(moveit_msgs::msg::RobotTrajectory(), request.getGroupName(), false);
            return response;
        }

        // TOTG alone is only velocity/acceleration-limited (no jerk bound), so
        // acceleration can still change instantaneously at corners between
        // waypoints. Ruckig smoothing runs on the already time-parameterized
        // trajectory and re-adjusts timing so jerk stays bounded too - same
        // adapter this repo already configures for the OMPL pipeline
        // (ompl_planning.yaml's AddRuckigTrajectorySmoothing). Non-fatal on
        // failure: the TOTG-only trajectory above is already valid and used
        // as-is if this doesn't succeed.
        if (!trajectory_processing::RuckigSmoothing::applySmoothing(
                *rt, profile.velocity_scaling > 0.0 ? profile.velocity_scaling : 1.0,
                profile.acceleration_scaling > 0.0 ? profile.acceleration_scaling : 1.0)) {
            RCLCPP_WARN(node_->get_logger(),
                "[MoveItCppPlannerManager] Ruckig jerk-limited smoothing failed for joint sequence - "
                "using TOTG-only timing instead.");
        }

        response.success = true;
        rt->getRobotTrajectoryMsg(response.trajectory);
        response.planning_time = 0.01;

        if (save_plan) {
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
        }

        publish_target_marker_for_request(request, true);
        publish_trajectory_markers(response.trajectory, request.getGroupName(), true);

        moveit_msgs::msg::DisplayTrajectory display_msg;
        display_msg.model_id = moveit_cpp_->getRobotModel()->getName();
        display_msg.trajectory.push_back(response.trajectory);
        moveit::core::robotStateToRobotStateMsg(*start_state, display_msg.trajectory_start);
        if (display_pub_) {
            display_pub_->publish(display_msg);
        }

        return response;
    } else if (!request.getWaypoints().empty()) {
        std::string ee_link = ee_link_for_group(request.getGroupName());
        if (ee_link.empty()) {
            response.success = false;
            response.error_message = "Could not resolve end-effector link for group " + request.getGroupName();
            publish_target_marker_for_request(request, false);
            publish_trajectory_markers(moveit_msgs::msg::RobotTrajectory(), request.getGroupName(), false);
            return response;
        }

        auto jmg = moveit_cpp_->getRobotModel()->getJointModelGroup(request.getGroupName());
        if (!jmg) {
            response.success = false;
            response.error_message = "Could not find joint model group " + request.getGroupName();
            publish_target_marker_for_request(request, false);
            publish_trajectory_markers(moveit_msgs::msg::RobotTrajectory(), request.getGroupName(), false);
            return response;
        }

        auto link_model = moveit_cpp_->getRobotModel()->getLinkModel(ee_link);
        if (!link_model) {
            response.success = false;
            response.error_message = "Could not find link model " + ee_link;
            publish_target_marker_for_request(request, false);
            publish_trajectory_markers(moveit_msgs::msg::RobotTrajectory(), request.getGroupName(), false);
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
            publish_target_marker_for_request(request, false);
            publish_trajectory_markers(moveit_msgs::msg::RobotTrajectory(), request.getGroupName(), false);
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

        double fraction = 0.0;
        bool planning_failed = false;
        size_t failed_step_idx = 0;
        double failed_segment_fraction = 0.0;
        moveit::core::RobotState active_state(*start_state);

        for (size_t i = 0; i < eigen_waypoints.size(); ++i) {
            std::vector<moveit::core::RobotStatePtr> segment_states;
            EigenSTL::vector_Isometry3d segment_waypoints = { eigen_waypoints[i] };
            
            double seg_fraction = moveit::core::CartesianInterpolator::computeCartesianPath(
                &active_state,
                jmg,
                segment_states,
                link_model,
                segment_waypoints,
                true,
                max_step,
                precision
            );

            if (seg_fraction < 0.99 || segment_states.empty()) {
                RCLCPP_WARN(node_->get_logger(),
                            "[MoveItCppPlannerManager] Cartesian planning failed for step %zu. Trying OMPL fallback...", i);
                
                planning_component->setStartState(active_state);
                
                std::string planning_frame = moveit_cpp_->getRobotModel()->getModelFrame();
                geometry_msgs::msg::PoseStamped target_pose;
                target_pose.header.frame_id = planning_frame;
                target_pose.pose.position.x = eigen_waypoints[i].translation().x();
                target_pose.pose.position.y = eigen_waypoints[i].translation().y();
                target_pose.pose.position.z = eigen_waypoints[i].translation().z();
                Eigen::Quaterniond q(eigen_waypoints[i].linear());
                target_pose.pose.orientation.x = q.x();
                target_pose.pose.orientation.y = q.y();
                target_pose.pose.orientation.z = q.z();
                target_pose.pose.orientation.w = q.w();
                planning_component->setGoal(target_pose, ee_link);

                moveit_cpp::PlanningComponent::PlanRequestParameters fallback_params = plan_params;
                fallback_params.planning_pipeline = "ompl";
                fallback_params.planner_id = "RRTConnectkConfigDefault";
                fallback_params.planning_attempts = 5;
                fallback_params.planning_time = 1.5;

                auto fallback_solution = planning_component->plan(fallback_params);
                if (fallback_solution) {
                    RCLCPP_INFO(node_->get_logger(),
                                "[MoveItCppPlannerManager] OMPL fallback succeeded for step %zu!", i);
                    
                    segment_states.clear();
                    for (size_t j = 0; j < fallback_solution.trajectory->getWayPointCount(); ++j) {
                        segment_states.push_back(std::make_shared<moveit::core::RobotState>(
                            fallback_solution.trajectory->getWayPoint(j)));
                    }
                    seg_fraction = 1.0;
                } else {
                    RCLCPP_ERROR(node_->get_logger(),
                                 "[MoveItCppPlannerManager] OMPL fallback also failed for step %zu", i);
                }
            }

            if (seg_fraction < 0.99 || segment_states.empty()) {
                planning_failed = true;
                failed_step_idx = i;
                failed_segment_fraction = seg_fraction;
                fraction = static_cast<double>(i) / eigen_waypoints.size();
                break;
            }

            traj_states.insert(traj_states.end(), segment_states.begin(), segment_states.end());
            active_state = *segment_states.back();
        }

        if (!planning_failed) {
            fraction = 1.0;
        }

        if (planning_failed || traj_states.empty()) {
            // Diagnostics for the failed step
            RCLCPP_ERROR(node_->get_logger(),
                         "[MoveItCppPlannerManager] Pre-planning failed at step %zu of %zu (segment fraction: %f, overall fraction: %f)",
                         failed_step_idx, eigen_waypoints.size(), failed_segment_fraction, fraction);

            const Eigen::Isometry3d& start_ee_pose = start_state->getGlobalLinkTransform(link_model);
            Eigen::Vector3d start_trans = start_ee_pose.translation();
            Eigen::Quaterniond start_rot(start_ee_pose.linear());
            RCLCPP_INFO(node_->get_logger(), "[PlannerDiagnostics] Start EE Pose: pos=[%.4f, %.4f, %.4f], q=[%.4f, %.4f, %.4f, %.4f]",
                        start_trans.x(), start_trans.y(), start_trans.z(),
                        start_rot.x(), start_rot.y(), start_rot.z(), start_rot.w());

            if (failed_step_idx < eigen_waypoints.size()) {
                Eigen::Vector3d wp_trans = eigen_waypoints[failed_step_idx].translation();
                Eigen::Quaterniond wp_rot(eigen_waypoints[failed_step_idx].linear());
                RCLCPP_INFO(node_->get_logger(), "[PlannerDiagnostics] Waypoint %zu Target: pos=[%.4f, %.4f, %.4f], q=[%.4f, %.4f, %.4f, %.4f]",
                            failed_step_idx, wp_trans.x(), wp_trans.y(), wp_trans.z(),
                            wp_rot.x(), wp_rot.y(), wp_rot.z(), wp_rot.w());
                
                moveit::core::RobotState test_state(active_state);
                bool ik_success = test_state.setFromIK(jmg, eigen_waypoints[failed_step_idx], ee_link);
                RCLCPP_INFO(node_->get_logger(), "[PlannerDiagnostics] IK solution for failing waypoint: %s", ik_success ? "SUCCESS" : "FAILED");
            }

            if (moveit_cpp_->getPlanningSceneMonitorNonConst()) {
                planning_scene_monitor::LockedPlanningSceneRO planning_scene(moveit_cpp_->getPlanningSceneMonitorNonConst());
                if (planning_scene) {
                    collision_detection::CollisionRequest coll_req;
                    coll_req.contacts = true;
                    coll_req.max_contacts = 10;
                    collision_detection::CollisionResult coll_res;
                    planning_scene->checkCollision(coll_req, coll_res, active_state);
                    RCLCPP_INFO(node_->get_logger(), "[PlannerDiagnostics] State before failure in collision: %s", coll_res.collision ? "YES" : "NO");
                    if (coll_res.collision) {
                        for (const auto& contact : coll_res.contacts) {
                            RCLCPP_INFO(node_->get_logger(), "[PlannerDiagnostics]   Collision contact between: %s and %s", 
                                        contact.first.first.c_str(), contact.first.second.c_str());
                        }
                    }
                }
            }

            response.success = false;
            response.error_message = "Cartesian planning failed at step " + std::to_string(failed_step_idx) +
                                     " (segment fraction: " + std::to_string(failed_segment_fraction) + " < 0.99)";
            publish_target_marker_for_request(request, false);
            publish_trajectory_markers(moveit_msgs::msg::RobotTrajectory(), request.getGroupName(), false);
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
            publish_target_marker_for_request(request, false);
            publish_trajectory_markers(moveit_msgs::msg::RobotTrajectory(), request.getGroupName(), false);
            return response;
        }

        response.success = true;
        rt->getRobotTrajectoryMsg(response.trajectory);
        response.planning_time = 0.01;

        // Save successfully run Cartesian plan
        if (save_plan) {
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
        }

        publish_target_marker_for_request(request, true);
        publish_trajectory_markers(response.trajectory, request.getGroupName(), true);

        // Publish display trajectory
        moveit_msgs::msg::DisplayTrajectory display_msg;
        display_msg.model_id = moveit_cpp_->getRobotModel()->getName();
        display_msg.trajectory.push_back(response.trajectory);
        moveit::core::RobotStatePtr start_state_ptr;
        if (request.getStartState()) {
            start_state_ptr = std::make_shared<moveit::core::RobotState>(*request.getStartState());
        } else {
            start_state_ptr = moveit_cpp_->getCurrentState();
        }
        if (start_state_ptr) {
            moveit::core::robotStateToRobotStateMsg(*start_state_ptr, display_msg.trajectory_start);
        }
        if (display_pub_) {
            display_pub_->publish(display_msg);
        }

        return response;
    } else {
        response.success = false;
        response.error_message = "No valid goal (pose or joints) specified in request.";
        publish_target_marker_for_request(request, false);
        publish_trajectory_markers(moveit_msgs::msg::RobotTrajectory(), request.getGroupName(), false);
        return response;
    }

    // 7. Perform planning
    auto plan_solution = planning_component->plan(plan_params);
    if (plan_solution) {
        response.success = true;
        plan_solution.trajectory->getRobotTrajectoryMsg(response.trajectory);
        response.planning_time = plan_solution.planning_time;

        // Save successfully run standard plan
        if (save_plan) {
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
        }
    } else {
        response.success = false;
        response.error_message = "Planning failed for group " + request.getGroupName();
    }

    publish_target_marker_for_request(request, response.success);
    publish_trajectory_markers(response.trajectory, request.getGroupName(), response.success);

    if (response.success) {
        moveit_msgs::msg::DisplayTrajectory display_msg;
        display_msg.model_id = moveit_cpp_->getRobotModel()->getName();
        display_msg.trajectory.push_back(response.trajectory);
        moveit::core::RobotStatePtr start_state_ptr = request.getStartState() ? 
            std::make_shared<moveit::core::RobotState>(*request.getStartState()) : 
            moveit_cpp_->getCurrentState();
        if (start_state_ptr) {
            moveit::core::robotStateToRobotStateMsg(*start_state_ptr, display_msg.trajectory_start);
        }
        if (display_pub_) {
            display_pub_->publish(display_msg);
        }
    }

    return response;
}

void MoveItCppPlannerManager::publish_target_marker_for_request(const planning_interface::PlannerRequest& request, bool visible)
{
    if (!target_marker_pub_) return;

    visualization_msgs::msg::Marker marker;
    marker.ns = "planning_target";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;

    if (!visible) {
        marker.header.frame_id = "world";
        marker.action = visualization_msgs::msg::Marker::DELETE;
        target_marker_pub_->publish(marker);
        return;
    }

    std::optional<geometry_msgs::msg::PoseStamped> target_pose;
    if (request.getTargetPose().has_value()) {
        target_pose = request.getTargetPose().value();
    } else if (!request.getWaypoints().empty()) {
        target_pose = request.getWaypoints().back();
    } else if (!request.getJointTargets().empty()) {
        if (moveit_cpp_ && moveit_cpp_->getRobotModel()) {
            moveit::core::RobotState goal_state(moveit_cpp_->getRobotModel());
            goal_state.setToDefaultValues();
            goal_state.setJointGroupPositions(request.getGroupName(), request.getJointTargets());
            goal_state.update();

            std::string ee_link = ee_link_for_group(request.getGroupName());
            if (!ee_link.empty()) {
                const Eigen::Isometry3d& ee_state = goal_state.getGlobalLinkTransform(ee_link);
                geometry_msgs::msg::PoseStamped pose;
                pose.header.frame_id = moveit_cpp_->getRobotModel()->getModelFrame();
                pose.header.stamp = node_->get_clock()->now();

                Eigen::Vector3d translation = ee_state.translation();
                Eigen::Quaterniond rotation(ee_state.linear());
                pose.pose.position.x = translation.x();
                pose.pose.position.y = translation.y();
                pose.pose.position.z = translation.z();
                pose.pose.orientation.x = rotation.x();
                pose.pose.orientation.y = rotation.y();
                pose.pose.orientation.z = rotation.z();
                pose.pose.orientation.w = rotation.w();

                target_pose = pose;
            }
        }
    } else if (!request.getJointSequence().empty()) {
        if (moveit_cpp_ && moveit_cpp_->getRobotModel()) {
            moveit::core::RobotState goal_state(moveit_cpp_->getRobotModel());
            goal_state.setToDefaultValues();
            goal_state.setJointGroupPositions(request.getGroupName(), request.getJointSequence().back());
            goal_state.update();

            std::string ee_link = ee_link_for_group(request.getGroupName());
            if (!ee_link.empty()) {
                const Eigen::Isometry3d& ee_state = goal_state.getGlobalLinkTransform(ee_link);
                geometry_msgs::msg::PoseStamped pose;
                pose.header.frame_id = moveit_cpp_->getRobotModel()->getModelFrame();
                pose.header.stamp = node_->get_clock()->now();

                Eigen::Vector3d translation = ee_state.translation();
                Eigen::Quaterniond rotation(ee_state.linear());
                pose.pose.position.x = translation.x();
                pose.pose.position.y = translation.y();
                pose.pose.position.z = translation.z();
                pose.pose.orientation.x = rotation.x();
                pose.pose.orientation.y = rotation.y();
                pose.pose.orientation.z = rotation.z();
                pose.pose.orientation.w = rotation.w();

                target_pose = pose;
            }
        }
    }

    if (target_pose.has_value()) {
        marker.header = target_pose.value().header;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose = target_pose.value().pose;
        marker.scale.x = 0.05; // 5 cm sphere
        marker.scale.y = 0.05;
        marker.scale.z = 0.05;
        // Sleek green color for successful plan target
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 0.8f; // slightly transparent
        
        target_marker_pub_->publish(marker);
    } else {
        marker.header.frame_id = "world";
        marker.action = visualization_msgs::msg::Marker::DELETE;
        target_marker_pub_->publish(marker);
    }
}

void MoveItCppPlannerManager::publish_trajectory_markers(const moveit_msgs::msg::RobotTrajectory& trajectory, const std::string& group, bool visible)
{
    if (!trajectory_marker_pub_) return;

    visualization_msgs::msg::Marker spheres_marker;
    spheres_marker.header.frame_id = moveit_cpp_->getRobotModel()->getModelFrame();
    spheres_marker.header.stamp = node_->get_clock()->now();
    spheres_marker.ns = "planned_trajectory";
    spheres_marker.id = 0;
    spheres_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;

    visualization_msgs::msg::Marker line_marker;
    line_marker.header = spheres_marker.header;
    line_marker.ns = "planned_trajectory";
    line_marker.id = 1;
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;

    if (!visible || trajectory.joint_trajectory.points.empty()) {
        spheres_marker.action = visualization_msgs::msg::Marker::DELETE;
        line_marker.action = visualization_msgs::msg::Marker::DELETE;
        trajectory_marker_pub_->publish(spheres_marker);
        trajectory_marker_pub_->publish(line_marker);
        return;
    }

    std::string ee_link = ee_link_for_group(group);
    if (ee_link.empty()) return;

    spheres_marker.action = visualization_msgs::msg::Marker::ADD;
    line_marker.action = visualization_msgs::msg::Marker::ADD;

    // Green sphere list for waypoints
    spheres_marker.scale.x = 0.02; // 2 cm diameter dots
    spheres_marker.scale.y = 0.02;
    spheres_marker.scale.z = 0.02;
    spheres_marker.color.r = 0.0f;
    spheres_marker.color.g = 1.0f;
    spheres_marker.color.b = 0.0f;
    spheres_marker.color.a = 0.8f;

    // Green line strip to connect waypoints
    line_marker.scale.x = 0.005; // 5 mm thick line
    line_marker.color.r = 0.0f;
    line_marker.color.g = 0.8f;
    line_marker.color.b = 0.0f;
    line_marker.color.a = 0.6f;

    moveit::core::RobotState state(moveit_cpp_->getRobotModel());
    
    for (const auto& point : trajectory.joint_trajectory.points) {
        state.setVariablePositions(trajectory.joint_trajectory.joint_names, point.positions);
        state.update();
        const Eigen::Isometry3d& ee_state = state.getGlobalLinkTransform(ee_link);
        Eigen::Vector3d translation = ee_state.translation();

        geometry_msgs::msg::Point pt;
        pt.x = translation.x();
        pt.y = translation.y();
        pt.z = translation.z();

        spheres_marker.points.push_back(pt);
        line_marker.points.push_back(pt);
    }

    trajectory_marker_pub_->publish(spheres_marker);
    trajectory_marker_pub_->publish(line_marker);
}

} // namespace motion_planner
