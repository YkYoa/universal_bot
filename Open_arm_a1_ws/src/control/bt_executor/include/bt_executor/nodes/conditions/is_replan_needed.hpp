#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// is_replan_needed.hpp
//
// Condition: returns SUCCESS (triggers recovery path) when either:
//   a) BB_REPLAN_NEEDED flag is true (set externally, e.g. by the VLA bridge),
//   b) BB_VLA_CONFIDENCE drops below the configured threshold.
//
// XML usage:
//   <IsReplanNeeded confidence_threshold="0.65" />
// ─────────────────────────────────────────────────────────────────────────────
#include "behaviortree_cpp/behavior_tree.h"
#include "bt_executor/blackboard_keys.hpp"

namespace bt_executor {

class IsReplanNeeded : public BT::ConditionNode
{
public:
  IsReplanNeeded(const std::string & name, const BT::NodeConfig & config)
  : BT::ConditionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<double>(
        "confidence_threshold",
        REPLAN_CONFIDENCE_THRESHOLD,
        "VLA confidence below which a replan is triggered")
    };
  }

  BT::NodeStatus tick() override
  {
    // Check explicit replan flag (set by /bt_executor/replan service)
    bool flag = false;
    (void)config().blackboard->get(BB_REPLAN_NEEDED, flag);
    if (flag) {
      // Consume the flag
      config().blackboard->set(BB_REPLAN_NEEDED, false);
      return BT::NodeStatus::SUCCESS;
    }

    // Check confidence gate
    double threshold = REPLAN_CONFIDENCE_THRESHOLD;
    getInput("confidence_threshold", threshold);

    double confidence = 1.0;
    (void)config().blackboard->get(BB_VLA_CONFIDENCE, confidence);

    return (confidence < threshold)
      ? BT::NodeStatus::SUCCESS
      : BT::NodeStatus::FAILURE;
  }
};

}  // namespace bt_executor
