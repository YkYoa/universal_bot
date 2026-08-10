#pragma once
// -----------------------------------------------------------------------------
// executor_app.hpp
//
// The shared main() body: bring up the node, wait for controllers, build the
// supervisor, spin.
//
// It exists so a project package can have its own executable - registering its
// hardcoded actions and pointing at its own sequence store - without copying
// forty lines of boilerplate. builds/qvic_2026/src/qvic_fsm_node.cpp is the
// whole reason, and is about a dozen lines because of this.
//
// Executor choice: single-threaded, deliberately. Every callback the FSM
// depends on - action results, service handlers, wait timers - is then
// delivered on one thread, which is what lets SequenceFsm hold mutable state
// without a single lock. Switching to a MultiThreadedExecutor means auditing
// that class for races first.
// -----------------------------------------------------------------------------
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "sequence_executor/builtin_actions.hpp"
#include "sequence_executor/sequence_source.hpp"

namespace sequence_executor {

// Called once, after the node exists and its parameters are declared, to
// supply the sequence source and register builtin actions. Returning nullptr
// falls back to a YamlSequenceSource built from the `sequence_yaml_path`
// parameter.
using ConfigureCallback = std::function<std::shared_ptr<SequenceSource>(
  const rclcpp::Node::SharedPtr& node, BuiltinActionRegistry& builtins)>;

// Declares these parameters on the node:
//   sequence_yaml_path     - fallback source; required if `configure` returns null
//   sequence_name          - autostart this sequence, empty = sit in IDLE
//   hardware_config_path   - override for the control-mode probe
//   wait_for_controllers   - seconds; 0 skips the wait entirely
int runApp(int argc, char** argv, const std::string& node_name,
           const ConfigureCallback& configure = nullptr);

}  // namespace sequence_executor
