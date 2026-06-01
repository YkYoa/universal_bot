#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// is_gripper_holding.hpp
//
// Condition: returns SUCCESS if the specified gripper is currently holding
// an object (force exceeds GRIP_FORCE_THRESHOLD or encoder reading).
//
// Blackboard reads:
//   BB_LEFT_GRIP_HOLDING  / BB_RIGHT_GRIP_HOLDING   bool
//   BB_LEFT_GRIP_FORCE    / BB_RIGHT_GRIP_FORCE      double [N]
//
// XML usage:
//   <IsGripperHolding arm="left_arm" />
// ─────────────────────────────────────────────────────────────────────────────
#include "behaviortree_cpp/behavior_tree.h"
#include "bt_executor/blackboard_keys.hpp"

namespace bt_executor {

class IsGripperHolding : public BT::ConditionNode
{
public:
  IsGripperHolding(const std::string & name, const BT::NodeConfig & config)
  : BT::ConditionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::string>("arm",
        "left_arm | right_arm — which gripper to check")
    };
  }

  BT::NodeStatus tick() override
  {
    // STUB: fake hardware has no force sensor.  Always assume grasp succeeded.
    // TODO: read actual gripper force/position feedback on real hardware.
    return BT::NodeStatus::SUCCESS;

    // Original implementation (requires real sensor data on blackboard):
    // std::string arm;
    // getInput("arm", arm);
    // const bool is_left = (arm == "left_arm");
    // const char* hold_key = is_left ? BB_LEFT_GRIP_HOLDING : BB_RIGHT_GRIP_HOLDING;
    // bool holding = false;
    // config().blackboard->get(hold_key, holding);
    // return holding ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }
};

}  // namespace bt_executor
