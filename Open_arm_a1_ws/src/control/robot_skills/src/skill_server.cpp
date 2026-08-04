#include "robot_skills/skill_server.hpp"
#include <moveit/trajectory_execution_manager/trajectory_execution_manager.hpp>

namespace robot_skills
{

SkillServer::SkillServer(const std::shared_ptr<rclcpp::Node>& node,
                         const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner)
  : node_(node), planner_(planner)
{
}

SkillServer::~SkillServer()
{
    execution_active_ = false;
    // Join both long-lived threads before destruction to avoid use-after-free.
    // execute_thread_ captures 'this' so it MUST be joined before the object dies.
    if (execute_thread_.joinable()) {
        execute_thread_.join();
    }
    if (replan_thread_.joinable()) {
        replan_thread_.join();
    }
}

bool SkillServer::start()
{
    using namespace std::placeholders;

    action_server_ = rclcpp_action::create_server<ExecuteSkill>(
        node_,
        "robot_skills_server/execute_skill",
        std::bind(&SkillServer::handle_goal, this, _1, _2),
        std::bind(&SkillServer::handle_cancel, this, _1),
        std::bind(&SkillServer::handle_accepted, this, _1)
    );

    RCLCPP_INFO(node_->get_logger(), "Robot Skills action server started.");
    return true;
}

void SkillServer::register_skill(const std::shared_ptr<RobotSkill>& skill)
{
    if (skill) {
        skill->initialize(node_, planner_);
        skills_[skill->name()] = skill;
        RCLCPP_INFO(node_->get_logger(), "Registered skill: %s", skill->name().c_str());
    }
}

rclcpp_action::GoalResponse SkillServer::handle_goal(
    const rclcpp_action::GoalUUID& /*uuid*/,
    std::shared_ptr<const ExecuteSkill::Goal> goal)
{
    RCLCPP_INFO(node_->get_logger(), "Received request for skill: %s on arm: %s", 
                goal->skill_name.c_str(), goal->arm.c_str());

    auto it = skills_.find(goal->skill_name);
    if (it == skills_.end()) {
        RCLCPP_WARN(node_->get_logger(), "Rejecting goal: skill '%s' not registered", goal->skill_name.c_str());
        return rclcpp_action::GoalResponse::REJECT;
    }

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse SkillServer::handle_cancel(
    const std::shared_ptr<GoalHandleExecuteSkill>& /*goal_handle*/)
{
    RCLCPP_INFO(node_->get_logger(), "Received cancel request.");
    execution_active_ = false;

    // Stop MoveItCpp execution
    if (planner_ && planner_->getMoveItCpp()) {
        auto tem = planner_->getMoveItCpp()->getTrajectoryExecutionManagerNonConst();
        if (tem) {
            tem->stopExecution(true);
        }
    }

    return rclcpp_action::CancelResponse::ACCEPT;
}

void SkillServer::handle_accepted(const std::shared_ptr<GoalHandleExecuteSkill>& goal_handle)
{
    // Join any previously finished execute thread before starting a new one.
    // We do NOT detach because the lambda captures 'this', and a detached thread
    // would be a use-after-free if the node shuts down while executing.
    if (execute_thread_.joinable()) {
        execute_thread_.join();
    }
    execute_thread_ = std::thread([this, goal_handle]() { execute_goal(goal_handle); });
}

void SkillServer::execute_goal(const std::shared_ptr<GoalHandleExecuteSkill>& goal_handle)
{
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<ExecuteSkill::Feedback>();
    auto result = std::make_shared<ExecuteSkill::Result>();

    auto it = skills_.find(goal->skill_name);
    if (it == skills_.end()) {
        result->success = false;
        result->error_message = "Skill not registered.";
        goal_handle->abort(result);
        return;
    }

    // Populate skill request
    SkillRequest req;
    req.arm = goal->arm;
    req.planner_profile = goal->planner_profile;
    req.mode = (goal->planning_mode == "realtime") ? PlanningMode::REALTIME : PlanningMode::NORMAL;
    req.target_pose = goal->target_pose;
    req.named_pose = goal->named_pose;
    req.waypoints = goal->waypoints;
    req.velocity_override = goal->velocity_override;
    req.acceleration_override = goal->acceleration_override;
    req.position_only = goal->position_only;
    req.joint_targets = goal->joint_targets;
    req.joint_sequence = goal->joint_sequence;

    feedback->status = "planning";
    goal_handle->publish_feedback(feedback);

    // Execute the skill
    auto skill_res = it->second->execute(req, goal_handle);

    result->success = skill_res.success;
    result->error_message = skill_res.error_message;
    result->planning_time_sec = skill_res.planning_time_sec;
    result->execution_time_sec = skill_res.execution_time_sec;

    if (skill_res.success) {
        RCLCPP_INFO(node_->get_logger(), "Skill '%s' completed successfully.", goal->skill_name.c_str());
        goal_handle->succeed(result);
    } else {
        RCLCPP_WARN(node_->get_logger(), "Skill '%s' failed: %s", goal->skill_name.c_str(), skill_res.error_message.c_str());
        goal_handle->abort(result);
    }
}

SkillResult SkillServer::execute_trajectory(
    const std::string& group_name,
    const moveit_msgs::msg::RobotTrajectory& trajectory,
    PlanningMode mode,
    const std::shared_ptr<GoalHandleExecuteSkill>& goal_handle,
    const SkillRequest& original_req)
{
    SkillResult result;
    auto feedback = std::make_shared<ExecuteSkill::Feedback>();

    if (!planner_ || !planner_->getMoveItCpp()) {
        result.success = false;
        result.error_message = "MoveItCpp is not initialized.";
        return result;
    }

    auto start_time = node_->now();

    if (mode == PlanningMode::NORMAL) {
        feedback->status = "executing";
        feedback->progress = 0.5f;
        goal_handle->publish_feedback(feedback);

        // Blocking execution via TrajectoryExecutionManager
        auto tem = planner_->getMoveItCpp()->getTrajectoryExecutionManagerNonConst();
        if (!tem) {
            result.success = false;
            result.error_message = "TrajectoryExecutionManager is not available.";
            return result;
        }

        tem->push(trajectory);
        auto status = tem->executeAndWait();
        bool execute_success = (status == moveit_controller_manager::ExecutionStatus::SUCCEEDED);
        
        auto end_time = node_->now();
        result.execution_time_sec = (end_time - start_time).seconds();
        
        if (execute_success) {
            result.success = true;
            feedback->progress = 1.0f;
            feedback->status = "done";
            goal_handle->publish_feedback(feedback);
        } else {
            result.success = false;
            result.error_message = "Blocking execution failed with status: " + status.asString();
        }
        return result;
    }

    // REALTIME mode: execute immediately and monitor goal modifications asynchronously
    execution_active_ = true;
    goal_changed_ = false;

    {
        std::lock_guard<std::mutex> lock(req_mutex_);
        updated_request_ = original_req;
    }

    feedback->status = "executing_realtime";
    feedback->progress = 0.1f;
    goal_handle->publish_feedback(feedback);

    // Non-blocking start via TrajectoryExecutionManager
    auto tem = planner_->getMoveItCpp()->getTrajectoryExecutionManagerNonConst();
    if (!tem) {
        execution_active_ = false;
        result.success = false;
        result.error_message = "TrajectoryExecutionManager is not available.";
        return result;
    }

    bool initial_exec = tem->push(trajectory);
    if (!initial_exec) {
        execution_active_ = false;
        result.success = false;
        result.error_message = "Failed to push initial trajectory.";
        return result;
    }
    tem->execute();

    // Create a subscription or service callback check, or spin a background replanning loop
    if (replan_thread_.joinable()) {
        replan_thread_.join();
    }

    replan_thread_ = std::thread([this, group_name, goal_handle, start_time]() {
        auto inner_feedback = std::make_shared<ExecuteSkill::Feedback>();
        
        while (execution_active_) {
            if (goal_changed_) {
                RCLCPP_INFO(node_->get_logger(), "[SkillServer] Goal changed mid-motion. Replanning...");
                
                inner_feedback->status = "replanning";
                goal_handle->publish_feedback(inner_feedback);

                SkillRequest current_req;
                {
                    std::lock_guard<std::mutex> lock(req_mutex_);
                    current_req = updated_request_;
                }

                // Stop the active trajectory first
                auto tem_stop = planner_->getMoveItCpp()->getTrajectoryExecutionManagerNonConst();
                if (tem_stop) {
                    tem_stop->stopExecution(true);
                }

                // Build a plan request
                planning_interface::PlannerRequest replan_req;
                replan_req.setGroupName(group_name);
                replan_req.setProfileName(current_req.planner_profile);
                
                if (current_req.target_pose.header.frame_id != "") {
                    replan_req.setTargetPose(current_req.target_pose);
                } else if (!current_req.named_pose.empty()) {
                    auto robot_model = planner_->getMoveItCpp()->getRobotModel();
                    if (robot_model) {
                        auto joint_model_group = robot_model->getJointModelGroup(group_name);
                        if (joint_model_group) {
                            std::map<std::string, double> joint_map;
                            if (joint_model_group->getVariableDefaultPositions(current_req.named_pose, joint_map)) {
                                std::vector<double> joint_values;
                                joint_values.reserve(joint_map.size());
                                for (const auto& name : joint_model_group->getVariableNames()) {
                                    joint_values.push_back(joint_map[name]);
                                }
                                replan_req.setJointTargets(joint_values);
                            }
                        }
                    }
                }

                auto plan_resp = planner_->plan(replan_req);
                if (plan_resp.success) {
                    RCLCPP_INFO(node_->get_logger(), "[SkillServer] Replan succeeded. Swapping trajectories.");
                    auto tem_swap = planner_->getMoveItCpp()->getTrajectoryExecutionManagerNonConst();
                    if (tem_swap) {
                        if (tem_swap->push(plan_resp.trajectory)) {
                            tem_swap->execute();
                        }
                    }
                    goal_changed_ = false;
                    
                    inner_feedback->status = "executing_realtime";
                    goal_handle->publish_feedback(inner_feedback);
                } else {
                    RCLCPP_WARN(node_->get_logger(), "[SkillServer] Replanning failed: %s", plan_resp.error_message.c_str());
                }
            }

            // Check if execution is finished using trajectory execution manager
            auto tem_check = planner_->getMoveItCpp()->getTrajectoryExecutionManagerNonConst();
            if (tem_check && tem_check->getLastExecutionStatus() != moveit_controller_manager::ExecutionStatus::RUNNING) {
                RCLCPP_INFO(node_->get_logger(), "[SkillServer] Trajectory execution finished.");
                execution_active_ = false;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    if (replan_thread_.joinable()) {
        replan_thread_.join();
    }

    auto end_time = node_->now();
    result.execution_time_sec = (end_time - start_time).seconds();

    // Check the actual final execution status — do NOT unconditionally return success.
    // If the replan failed or TEM stopped with an error, propagate that to the BT.
    auto tem_final = planner_->getMoveItCpp()->getTrajectoryExecutionManagerNonConst();
    if (tem_final) {
        auto final_status = tem_final->getLastExecutionStatus();
        if (final_status == moveit_controller_manager::ExecutionStatus::SUCCEEDED ||
            final_status == moveit_controller_manager::ExecutionStatus::RUNNING) {
            // RUNNING means TEM finished the trajectory and is idle (last status is RUNNING
            // when no trajectory is queued, which is normal post-completion).
            result.success = true;
        } else {
            result.success = false;
            result.error_message = "Realtime execution ended with status: " + final_status.asString();
            RCLCPP_WARN(node_->get_logger(),
                "[SkillServer] REALTIME execution finished with non-success status: %s",
                final_status.asString().c_str());
        }
    } else {
        // TEM unavailable post-execution — conservatively report failure.
        result.success = false;
        result.error_message = "TrajectoryExecutionManager unavailable after realtime execution.";
    }

    return result;
}

} // namespace robot_skills
