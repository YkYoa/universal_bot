#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// blackboard_keys.hpp
//
// Single source of truth for every key written/read on the shared BT
// blackboard.  Import this header in every node implementation so key
// names never drift apart between writers and readers.
//
// Naming convention:
//   - Input data written BY the VLA bridge:  vla_*
//   - Arm / gripper state from ros2_control:  arm_*, gripper_*
//   - Task-level status flags:                task_*
//   - Intermediate planning data:             plan_*
// ─────────────────────────────────────────────────────────────────────────────

namespace bt_executor {

// ── VLA layer writes these ────────────────────────────────────────────────────

/// geometry_msgs::msg::PoseStamped — target for the next grasp
static constexpr const char* BB_GRASP_POSE        = "vla_grasp_pose";

/// geometry_msgs::msg::PoseStamped — target for the next place
static constexpr const char* BB_PLACE_POSE        = "vla_place_pose";

/// std::string — e.g. "pick_and_place", "hand_over", "inspect"
static constexpr const char* BB_TASK_NAME         = "vla_task_name";

/// std::string — object class label from the detector ("bottle", "box", …)
static constexpr const char* BB_TARGET_OBJECT     = "vla_target_object";

/// double — VLM confidence in the current plan [0, 1]
static constexpr const char* BB_VLA_CONFIDENCE    = "vla_confidence";

/// bool — set true by VLA bridge when a new goal arrives
static constexpr const char* BB_NEW_GOAL_READY    = "vla_new_goal_ready";

// ── Arm / gripper state (written by state-monitor node) ──────────────────────

/// sensor_msgs::msg::JointState — latest from /joint_states
static constexpr const char* BB_JOINT_STATE       = "arm_joint_state";

/// double — left finger force [N], from /left_gripper_controller/state
static constexpr const char* BB_LEFT_GRIP_FORCE   = "gripper_left_force";

/// double — right finger force [N]
static constexpr const char* BB_RIGHT_GRIP_FORCE  = "gripper_right_force";

/// bool — true if left gripper detects contact above threshold
static constexpr const char* BB_LEFT_GRIP_HOLDING = "gripper_left_holding";

/// bool — true if right gripper detects contact above threshold
static constexpr const char* BB_RIGHT_GRIP_HOLDING = "gripper_right_holding";

// ── Task status flags (written by action nodes) ───────────────────────────────

/// bool — set by IsObjectVisible condition
static constexpr const char* BB_OBJECT_VISIBLE    = "task_object_visible";

/// bool — MoveIt reported the last plan as feasible
static constexpr const char* BB_PLAN_FEASIBLE     = "task_plan_feasible";

/// bool — triggers full replan via QueryVLA action node
static constexpr const char* BB_REPLAN_NEEDED     = "task_replan_needed";

/// int — incremented on each recovery attempt; reset on task success
static constexpr const char* BB_RECOVERY_COUNT    = "task_recovery_count";

/// std::string — human-readable status for UI telemetry
static constexpr const char* BB_STATUS_MSG        = "task_status_message";

// ── Planning data (intermediate, shared between action nodes) ─────────────────

/// std::string — "left_arm" | "right_arm" | "both_arms"
static constexpr const char* BB_ACTIVE_ARM        = "plan_active_arm";

/// double — velocity scaling factor forwarded to MoveIt [0.0, 1.0]
static constexpr const char* BB_VEL_SCALE         = "plan_velocity_scaling";

/// double — acceleration scaling factor [0.0, 1.0]
static constexpr const char* BB_ACC_SCALE         = "plan_accel_scaling";

// ── Planner pipeline selection (written by SetPlannerConfig node) ─────────────

/// std::string — MoveIt pipeline identifier (e.g. "ompl", "pilz_industrial_motion_planner")
static constexpr const char* BB_PIPELINE_ID       = "plan_pipeline_id";

/// std::string — MoveIt planner identifier within the pipeline (e.g. "RRTConnectkConfigDefault", "PTP")
static constexpr const char* BB_PLANNER_ID        = "plan_planner_id";

/// double — allowed planning time [s]
static constexpr const char* BB_PLAN_TIME         = "plan_planning_time";

/// int — number of planning attempts
static constexpr const char* BB_NUM_ATTEMPTS      = "plan_num_attempts";

/// std::string — active planner profile name (e.g. "safe_rrt", "linear_approach")
static constexpr const char* BB_PLANNER_PROFILE   = "plan_planner_profile";

/// moveit_msgs::msg::RobotTrajectory — planned trajectory ready for execution
static constexpr const char* BB_PLANNED_TRAJECTORY= "plan_trajectory";

// ── Realtime replanning (written by goal monitor / VLA bridge) ────────────────

/// bool — true when the target goal has changed and in-flight motion should replan
static constexpr const char* BB_GOAL_CHANGED      = "task_goal_changed";

/// int — monotonically increasing counter; bumped each time a new goal arrives
static constexpr const char* BB_GOAL_STAMP        = "task_goal_stamp";

// ── Constants ─────────────────────────────────────────────────────────────────

/// Force threshold (N) above which a gripper is considered "holding"
static constexpr double GRIP_FORCE_THRESHOLD = 2.0;

/// Joint-error tolerance (rad) for IsArmAtPose
static constexpr double POSE_TOLERANCE_RAD = 0.05;

/// VLA confidence below which a replan is triggered
static constexpr double REPLAN_CONFIDENCE_THRESHOLD = 0.65;

/// Max recovery attempts before SafeAbort
static constexpr int MAX_RECOVERY_ATTEMPTS = 3;

}  // namespace bt_executor
