#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// is_goal_changed.hpp
//
// Condition: returns SUCCESS if the target goal has changed since the last
// time this node consumed the change flag.  This enables reactive replanning
// when an external system (VLA bridge, human operator) updates the goal
// pose while the robot is already moving.
//
// The VLA bridge or REST API sets BB_GOAL_CHANGED=true and bumps
// BB_GOAL_STAMP whenever a new goal arrives.  This condition checks those
// keys and consumes the flag.
//
// Usage in a reactive BT pattern (ReactiveSequence):
//
//   <ReactiveSequence>
//     <Inverter>
//       <IsGoalChanged />           <!-- re-checks every tick -->
//     </Inverter>
//     <MoveToPose ... />            <!-- gets halted if goal changes -->
//   </ReactiveSequence>
//
//   When IsGoalChanged returns SUCCESS (goal changed), the Inverter
//   returns FAILURE, which halts MoveToPose and lets the parent
//   re-enter the approach sequence with the new goal.
//
// XML usage:
//   <IsGoalChanged />
// ─────────────────────────────────────────────────────────────────────────────
#include "behaviortree_cpp/behavior_tree.h"
#include "bt_executor/blackboard_keys.hpp"

namespace bt_executor {

class IsGoalChanged : public BT::ConditionNode
{
public:
  IsGoalChanged(const std::string & name, const BT::NodeConfig & config)
  : BT::ConditionNode(name, config), last_seen_stamp_(0) {}

  static BT::PortsList providedPorts()
  {
    return {};
  }

  BT::NodeStatus tick() override
  {
    // Check the explicit change flag
    bool changed = false;
    config().blackboard->get(BB_GOAL_CHANGED, changed);

    if (changed) {
      // Consume the flag so we don't re-trigger
      config().blackboard->set(BB_GOAL_CHANGED, false);
      return BT::NodeStatus::SUCCESS;  // Goal HAS changed
    }

    return BT::NodeStatus::FAILURE;  // Goal has NOT changed
  }

private:
  int last_seen_stamp_;
};

}  // namespace bt_executor
