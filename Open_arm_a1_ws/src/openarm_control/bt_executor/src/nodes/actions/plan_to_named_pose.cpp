// plan_to_named_pose.cpp
#include "bt_executor/nodes/actions/plan_to_named_pose.hpp"

namespace bt_executor {

// Mirror of SRDF group_states — keep in sync with openarm_bimanual.srdf
const std::map<std::string, std::map<std::string, std::vector<double>>>
PlanToNamedPose::named_poses_ = {
  {"left_arm", {
    {"home",  {0.0,  0.0, 0.0, -1.5, 0.0, 0.0, 0.0}},
    {"ready", {0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0}},
  }},
  {"right_arm", {
    {"home",  {0.0, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0}},
    {"ready", {0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0}},
  }},
};

BT::NodeStatus PlanToNamedPose::tick()
{
  std::string arm, pose_name;
  double duration = 3.0;
  getInput("arm",       arm);
  getInput("pose_name", pose_name);
  getInput("duration",  duration);

  // Lookup joint positions
  auto arm_it = named_poses_.find(arm);
  if (arm_it == named_poses_.end()) {
    std::cout << "[PlanToNamedPose] Unknown arm: " << arm << std::endl;
    return BT::NodeStatus::FAILURE;
  }
  auto pose_it = arm_it->second.find(pose_name);
  if (pose_it == arm_it->second.end()) {
    std::cout << "[PlanToNamedPose] Unknown pose: " << pose_name << std::endl;
    return BT::NodeStatus::FAILURE;
  }

  moveit_msgs::msg::RobotTrajectory traj;
  
  // Build joint names  e.g. openarm_left_joint1 … joint7
  const std::string pfx = (arm == "left_arm") ? "openarm_left_joint" : "openarm_right_joint";
  for (int i = 1; i <= 7; ++i) {
    traj.joint_trajectory.joint_names.push_back(pfx + std::to_string(i));
  }

  trajectory_msgs::msg::JointTrajectoryPoint pt;
  pt.positions = pose_it->second;
  pt.time_from_start.sec = static_cast<int32_t>(duration);
  pt.time_from_start.nanosec =
    static_cast<uint32_t>((duration - std::floor(duration)) * 1e9);
  
  traj.joint_trajectory.points.push_back(pt);

  setOutput("output_trajectory", traj);

  std::cout << "[PlanToNamedPose] " << arm << " -> " << pose_name 
            << " (" << duration << " s)" << std::endl;
  
  return BT::NodeStatus::SUCCESS;
}

}  // namespace bt_executor
