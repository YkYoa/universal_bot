// execute_trajectory.cpp
#include "bt_executor/nodes/actions/execute_trajectory.hpp"
#include "rclcpp/rclcpp.hpp"

namespace bt_executor {

static const char* execute_error_str(int code) {
  switch (code) {
    case 1:    return "SUCCESS";
    case -4:   return "CONTROL_FAILED";
    case -6:   return "TIMED_OUT";
    case -12:  return "GOAL_IN_COLLISION";
    case 99999:return "FAILURE";
    default:   return "UNKNOWN";
  }
}

bool ExecuteTrajectory::setGoal(Goal & goal)
{
  moveit_msgs::msg::RobotTrajectory traj;
  if (!getInput("trajectory", traj)) {
    RCLCPP_ERROR(logger(), "[ExecuteTrajectory] trajectory port not set");
    return false;
  }

  goal.trajectory = traj;
  RCLCPP_INFO(logger(), "[ExecuteTrajectory] Executing trajectory with %zu points", 
              traj.joint_trajectory.points.size());

  return true;
}

BT::NodeStatus ExecuteTrajectory::onResultReceived(const WrappedResult & result)
{
  const int code = result.result->error_code.val;
  if (code == 1 /* SUCCESS */) {
    RCLCPP_INFO(logger(), "[ExecuteTrajectory] success");
    return BT::NodeStatus::SUCCESS;
  }

  RCLCPP_WARN(logger(), "[ExecuteTrajectory] failed: %s (code %d)", execute_error_str(code), code);
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus ExecuteTrajectory::onFailure(BT::ActionNodeErrorCode error)
{
  RCLCPP_ERROR(logger(), "[ExecuteTrajectory] action failed: %s", BT::toStr(error));
  return BT::NodeStatus::FAILURE;
}

}  // namespace bt_executor
