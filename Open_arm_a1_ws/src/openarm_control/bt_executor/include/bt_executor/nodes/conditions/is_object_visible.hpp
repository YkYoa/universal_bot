#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// is_object_visible.hpp
//
// Condition: returns SUCCESS if the target object label is present in the
// world model (blackboard key BB_OBJECT_VISIBLE).
//
// Blackboard reads:
//   BB_OBJECT_VISIBLE   bool   written by the VLA perception bridge
//   BB_TARGET_OBJECT    string label to match (also a node input port)
//
// XML usage:
//   <IsObjectVisible object_label="{vla_target_object}" />
// ─────────────────────────────────────────────────────────────────────────────
#include "behaviortree_cpp/behavior_tree.h"
#include "bt_executor/blackboard_keys.hpp"

namespace bt_executor {

class IsObjectVisible : public BT::ConditionNode
{
public:
  IsObjectVisible(const std::string & name, const BT::NodeConfig & config)
  : BT::ConditionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::string>("object_label", "Object class label to check")
    };
  }

  BT::NodeStatus tick() override;
};

}  // namespace bt_executor
