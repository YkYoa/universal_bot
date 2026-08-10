#pragma once
// -----------------------------------------------------------------------------
// builtin_actions.hpp
//
// The hardcoded-action slot.
//
// Some behaviour is easier to write as C++ than to express as a step list -
// anything that branches on sensor state, or that a competition run needs to
// be exactly reproducible. Those live in the project package
// (builds/qvic_2026/src/qvic_actions.cpp) and register here.
//
// A registered action appears alongside stored sequences under the name
// "builtin:<id>", so the Android app, the web page, and the RunSequence action
// all reach it exactly the way they reach a stored sequence - no second code
// path, no second list to keep in sync.
//
// An action body is async, like everything else in this FSM: it gets a context
// full of clients and a done-callback to fire when it finishes. It must not
// block - the executor that would deliver its own results is the thread it
// would be blocking.
// -----------------------------------------------------------------------------
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "sequence_executor/hand_gripper_client.hpp"
#include "sequence_executor/scene_client.hpp"
#include "sequence_executor/sequence_source.hpp"
#include "sequence_executor/skill_client.hpp"

namespace sequence_executor {

// The prefix that marks a sequence name as a builtin rather than a stored one.
inline constexpr const char* kBuiltinPrefix = "builtin:";

struct BuiltinContext
{
  rclcpp::Node::SharedPtr node;
  std::shared_ptr<SkillClient> skill;
  std::shared_ptr<HandGripperClient> hand;
  std::shared_ptr<SceneClient> scene;
  std::shared_ptr<SequenceSource> source;

  // Poll between async hops. Returns true once the operator has cancelled or
  // e-stopped; an action that ignores it simply cannot be stopped early.
  std::function<bool()> cancelled;
};

struct BuiltinAction
{
  using DoneCallback = std::function<void(bool success, const std::string& error_message)>;

  std::string id;
  std::string label;
  std::string description;
  // One of kModeAny / kModeMotion / kModeTorque - checked before the body runs,
  // same as a stored step's.
  std::string required_control_mode;
  std::function<void(BuiltinContext&, DoneCallback)> run;
};

class BuiltinActionRegistry
{
public:
  // Later registrations of the same id replace earlier ones, so a project can
  // override a shared default without editing it.
  void add(BuiltinAction action);

  const BuiltinAction* find(const std::string& id) const;

  // Registration order, which is the order they appear in the UI.
  const std::vector<BuiltinAction>& all() const { return ordered_; }

  bool empty() const { return ordered_.empty(); }

  // Strips kBuiltinPrefix if present; returns empty for a stored-sequence name.
  static std::string idFromSequenceName(const std::string& name);

private:
  std::vector<BuiltinAction> ordered_;
  std::map<std::string, std::size_t> by_id_;
};

}  // namespace sequence_executor
