// plan_to_pose.cpp
#include "bt_executor/nodes/actions/plan_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"

namespace bt_executor {

// MoveIt error code → readable string (subset)
static const char* moveit_error_str(int code) {
  switch (code) {
    case 1:    return "SUCCESS";
    case -1:   return "PLANNING_FAILED";
    case -4:   return "CONTROL_FAILED";
    case -6:   return "TIMED_OUT";
    case -12:  return "GOAL_IN_COLLISION";
    case -31:  return "NO_IK_SOLUTION";
    case 99999:return "FAILURE";
    default:   return "UNKNOWN";
  }
}

bool PlanToPose::setGoal(Goal & goal)
{
  // ── Read ports ──────────────────────────────────────────────────────────
  std::string arm;
  getInput("arm", arm);

  geometry_msgs::msg::PoseStamped target;
  if (!getInput("target_pose", target)) {
    RCLCPP_ERROR(logger(), "[PlanToPose] target_pose port not set");
    return false;
  }

  double vel = 0.3, acc = 0.3, pos_tol = 0.01, orient_tol = 0.5;
  bool position_only = false;
  getInput("velocity_scaling",     vel);
  getInput("acceleration_scaling", acc);
  getInput("pos_tolerance",        pos_tol);
  getInput("orient_tolerance",     orient_tol);
  getInput("position_only",        position_only);

  // ── EE link from arm name ───────────────────────────────────────────────
  const std::string ee_link = (arm == "left_arm")
    ? "openarm_left_hand_tcp"
    : "openarm_right_hand_tcp";

  // ── Read planner config from blackboard (set by SetPlannerConfig) ───────
  // Falls back to safe defaults if no profile has been set.
  std::string pipeline_id = "ompl";
  std::string planner_id  = "RRTConnectkConfigDefault";
  double      plan_time   = 5.0;
  int         num_att     = 10;

  config().blackboard->get(BB_PIPELINE_ID,  pipeline_id);
  config().blackboard->get(BB_PLANNER_ID,   planner_id);
  config().blackboard->get(BB_PLAN_TIME,    plan_time);
  config().blackboard->get(BB_NUM_ATTEMPTS, num_att);

  // Override velocity/acceleration from blackboard if SetPlannerConfig set them
  config().blackboard->get(BB_VEL_SCALE, vel);
  config().blackboard->get(BB_ACC_SCALE, acc);

  // ── Build MoveGroup goal ────────────────────────────────────────────────
  goal.planning_options.plan_only = true;  // <--- KEY CHANGE: ONLY PLAN
  goal.planning_options.replan    = false; // Replanning handled by BT logic

  auto & req = goal.request;
  req.group_name                  = arm;
  req.num_planning_attempts       = num_att;
  req.allowed_planning_time       = plan_time;
  req.max_velocity_scaling_factor     = vel;
  req.max_acceleration_scaling_factor = acc;
  req.pipeline_id = pipeline_id;
  req.planner_id  = planner_id;

  req.workspace_parameters.header.frame_id = "world";
  req.workspace_parameters.min_corner.x = -2.0;
  req.workspace_parameters.min_corner.y = -2.0;
  req.workspace_parameters.min_corner.z = -2.0;
  req.workspace_parameters.max_corner.x =  2.0;
  req.workspace_parameters.max_corner.y =  2.0;
  req.workspace_parameters.max_corner.z =  2.0;

  // ── Position constraint ─────────────────────────────────────────────────
  moveit_msgs::msg::Constraints constraints;

  moveit_msgs::msg::PositionConstraint pc;
  pc.header.frame_id = "world";
  pc.link_name       = ee_link;
  pc.weight          = 1.0;

  shape_msgs::msg::SolidPrimitive sphere;
  sphere.type        = shape_msgs::msg::SolidPrimitive::SPHERE;
  sphere.dimensions  = {pos_tol};

  moveit_msgs::msg::BoundingVolume bv;
  bv.primitives.push_back(sphere);
  bv.primitive_poses.push_back(target.pose);
  pc.constraint_region = bv;

  constraints.position_constraints.push_back(pc);

  // ── Orientation constraint (skip if position_only) ──────────────────────
  if (!position_only) {
    moveit_msgs::msg::OrientationConstraint oc;
    oc.header.frame_id           = "world";
    oc.link_name                 = ee_link;
    oc.orientation               = target.pose.orientation;
    oc.absolute_x_axis_tolerance = orient_tol;
    oc.absolute_y_axis_tolerance = orient_tol;
    oc.absolute_z_axis_tolerance = orient_tol;
    oc.weight                    = 1.0;
    constraints.orientation_constraints.push_back(oc);
  }

  req.goal_constraints.push_back(constraints);

  RCLCPP_INFO(logger(),
    "[PlanToPose] %s → (%.3f, %.3f, %.3f) pipeline=%s planner=%s position_only=%s",
    arm.c_str(),
    target.pose.position.x, target.pose.position.y, target.pose.position.z,
    pipeline_id.c_str(), planner_id.c_str(),
    position_only ? "true" : "false");

  return true;
}

BT::NodeStatus PlanToPose::onResultReceived(const WrappedResult & result)
{
  const int code = result.result->error_code.val;
  if (code == 1 /* SUCCESS */) {
    // <--- KEY CHANGE: EXTRACT TRAJECTORY AND SAVE TO OUTPUT PORT
    setOutput("output_trajectory", result.result->planned_trajectory);
    
    config().blackboard->set(BB_STATUS_MSG, std::string("plan_to_pose: success"));
    return BT::NodeStatus::SUCCESS;
  }

  const std::string msg = std::string("plan_to_pose failed: ") + moveit_error_str(code);
  config().blackboard->set(BB_STATUS_MSG, msg);
  config().blackboard->set(BB_REPLAN_NEEDED, true);
  RCLCPP_WARN(logger(), "[PlanToPose] %s (code %d)", msg.c_str(), code);
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus PlanToPose::onFailure(BT::ActionNodeErrorCode error)
{
  RCLCPP_ERROR(logger(), "[PlanToPose] action failed: %s",
    BT::toStr(error));
  config().blackboard->set(BB_REPLAN_NEEDED, true);
  return BT::NodeStatus::FAILURE;
}

}  // namespace bt_executor
