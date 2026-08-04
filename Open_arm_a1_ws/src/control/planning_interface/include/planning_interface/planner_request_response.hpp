#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// planner_request_response.hpp
//
// PlannerRequest / PlannerResponse — the typed interface between the
// robot_skills layer and the motion_planner backend.
//
// Callers (robot_skills skills) build a PlannerRequest, pass it to
// PlannerManagerBase::plan(), and receive a PlannerResponse.
// ─────────────────────────────────────────────────────────────────────────────

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <optional>
#include <string>
#include <vector>
#include <memory>

namespace planning_interface {

// ── PlanRequestParameters ─────────────────────────────────────────────────────
// Explicit overrides; leave empty/zero to derive values from PlannerProfile.

struct PlanRequestParameters {
  std::string pipeline_id;         // overrides profile.pipeline_id if non-empty
  std::string planner_id;          // overrides profile.planner_id if non-empty
  int    planning_attempts = 0;    // 0 = use profile value
  double planning_time     = 0.0;  // 0 = use profile value
  double velocity_scaling  = 0.0;  // 0 = use profile value
  double acceleration_scaling = 0.0;
};

// ── PlannerRequest ────────────────────────────────────────────────────────────

class PlannerRequest {
public:
  PlannerRequest() = default;
  ~PlannerRequest() = default;

  // Planning group (e.g. "left_arm", "right_arm", "both_arms")
  const std::string & getGroupName() const   { return group_name_; }
  void setGroupName(const std::string & g)   { group_name_ = g; }

  // Profile name from planner_profiles.yaml (e.g. "safe_rrt", "linear_approach")
  const std::string & getProfileName() const { return profile_name_; }
  void setProfileName(const std::string & p) { profile_name_ = p; }

  // Cartesian EE goal — mutually exclusive with joint targets
  const std::optional<geometry_msgs::msg::PoseStamped> & getTargetPose() const {
    return target_pose_;
  }
  void setTargetPose(const geometry_msgs::msg::PoseStamped & pose) {
    target_pose_   = pose;
    joint_targets_.clear();
  }

  // Joint-space goal — mutually exclusive with pose target
  const std::vector<double> & getJointTargets() const { return joint_targets_; }
  void setJointTargets(const std::vector<double> & joints) {
    joint_targets_ = joints;
    target_pose_.reset();
  }

  // Optional explicit start state (nullptr = use current robot state)
  moveit::core::RobotStateConstPtr getStartState() const { return start_state_; }
  void setStartState(moveit::core::RobotStateConstPtr s) { start_state_ = s; }

  // Optional parameter overrides (fine-tuning on top of the profile)
  const PlanRequestParameters & getParameters() const   { return parameters_; }
  void setParameters(const PlanRequestParameters & p)   { parameters_ = p; }

  // Cartesian path waypoints (used by CartesianMoveSkill)
  const std::vector<geometry_msgs::msg::PoseStamped> & getWaypoints() const {
    return waypoints_;
  }
  void setWaypoints(const std::vector<geometry_msgs::msg::PoseStamped> & wps) {
    waypoints_ = wps;
  }

  // Joint-space waypoint sequence (used by MoveToJointSequenceSkill) -
  // mutually exclusive with pose target / single joint target. Each entry
  // is one waypoint's joint values.
  const std::vector<std::vector<double>> & getJointSequence() const {
    return joint_sequence_;
  }
  void setJointSequence(const std::vector<std::vector<double>> & seq) {
    joint_sequence_ = seq;
    target_pose_.reset();
    joint_targets_.clear();
  }

  // Position-only flag: skip orientation constraint in IK
  bool isPositionOnly() const          { return position_only_; }
  void setPositionOnly(bool p)         { position_only_ = p; }

private:
  std::string                                         group_name_;
  std::string                                         profile_name_;
  std::optional<geometry_msgs::msg::PoseStamped>      target_pose_;
  std::vector<double>                                 joint_targets_;
  std::vector<geometry_msgs::msg::PoseStamped>        waypoints_;
  std::vector<std::vector<double>>                    joint_sequence_;
  moveit::core::RobotStateConstPtr                    start_state_;
  PlanRequestParameters                               parameters_;
  bool                                                position_only_ = false;
};

// ── PlannerResponse ───────────────────────────────────────────────────────────

struct PlannerResponse {
  bool        success       = false;
  moveit_msgs::msg::RobotTrajectory trajectory;
  double      planning_time = 0.0;     // seconds
  std::string error_message;
};

}  // namespace planning_interface
