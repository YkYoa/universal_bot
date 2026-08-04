#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// is_arm_at_pose.hpp
//
// Condition: returns SUCCESS if the arm's current joint config is within
// POSE_TOLERANCE_RAD of the last commanded pose.
//
// Blackboard reads:
//   BB_JOINT_STATE   sensor_msgs::msg::JointState
//
// XML usage:
//   <IsArmAtPose arm="left_arm" tolerance="0.05" />
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include "behaviortree_cpp/behavior_tree.h"
#include "sensor_msgs/msg/joint_state.hpp"
#include "bt_executor/blackboard_keys.hpp"

namespace bt_executor {

class IsArmAtPose : public BT::ConditionNode
{
public:
  IsArmAtPose(const std::string & name, const BT::NodeConfig & config)
  : BT::ConditionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::string>("arm", "left_arm | right_arm"),
      BT::InputPort<double>("tolerance", POSE_TOLERANCE_RAD, "Joint error tolerance (rad)"),
      // The target positions are written to this port by MoveToPose after planning
      BT::InputPort<std::vector<double>>("target_positions", "Commanded joint positions")
    };
  }

  BT::NodeStatus tick() override
  {
    std::vector<double> targets;
    if (!getInput("target_positions", targets) || targets.empty()) {
      // No target recorded yet — assume not at pose
      return BT::NodeStatus::FAILURE;
    }

    double tolerance = POSE_TOLERANCE_RAD;
    getInput("tolerance", tolerance);

    sensor_msgs::msg::JointState js;
    if (!config().blackboard->get(BB_JOINT_STATE, js)) {
      return BT::NodeStatus::FAILURE;
    }

    std::string arm;
    getInput("arm", arm);
    const std::string prefix = (arm == "left_arm") ? "openarm_left_joint" : "openarm_right_joint";

    // Check each joint
    for (size_t i = 0; i < targets.size(); ++i) {
      const std::string joint_name = prefix + std::to_string(i + 1);
      auto it = std::find(js.name.begin(), js.name.end(), joint_name);
      if (it == js.name.end()) return BT::NodeStatus::FAILURE;
      const size_t idx = std::distance(js.name.begin(), it);
      if (std::abs(js.position[idx] - targets[i]) > tolerance) {
        return BT::NodeStatus::FAILURE;
      }
    }

    return BT::NodeStatus::SUCCESS;
  }
};

}  // namespace bt_executor
