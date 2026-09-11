#pragma once
// -----------------------------------------------------------------------------
// robot_supervisor.hpp
//
// The robot-level state machine, and the only ROS surface of this node.
//
// It sits above SequenceFsm and answers a different question: not "which step
// is running" but "what is this robot doing at all" - idle, running, paused,
// faulted, e-stopped, or being hand-guided. The Android app shows this; the
// sequence layer's step index is the detail underneath it.
//
// Exposes:
//   action  ~/run_sequence   (openarm_messages/action/RunSequence)
//   service ~/fsm_command    (openarm_messages/srv/FsmCommand)
//   topic   ~/state          (openarm_messages/msg/FsmState, latched)
//
// The topic is transient_local so a client that connects mid-run gets the
// current state immediately instead of waiting for the next transition. It is
// published on transitions only - there is no timer, so a robot sitting idle
// produces no traffic.
//
// One goal at a time: a RunSequence goal arriving while another is running is
// rejected rather than queued, because two sequences driving the same arm is
// never what the operator meant.
// -----------------------------------------------------------------------------
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <openarm_messages/action/run_sequence.hpp>
#include <openarm_messages/msg/fsm_state.hpp>
#include <openarm_messages/srv/fsm_command.hpp>

#include "sequence_executor/builtin_actions.hpp"
#include "sequence_executor/control_mode_probe.hpp"
#include "sequence_executor/sequence_fsm.hpp"
#include "sequence_executor/sequence_source.hpp"

namespace sequence_executor {

enum class RobotState
{
  BOOTING,    // waiting for the controllers to come up
  IDLE,
  RUNNING,
  PAUSED,
  FAULT,      // a sequence failed; needs clear_fault before anything else runs
  ESTOP,      // operator stopped everything; needs clear_fault
  TEACHING,   // hand-guiding, only reachable when the arm is in torque mode
};

/// Human-readable name of `state`, for logs and the FsmState message.
const char* toString(RobotState state);

/// The robot-level state machine and only ROS surface of this node - see
/// file header comment for the RunSequence/FsmCommand/FsmState surface and
/// why this is a distinct layer above SequenceFsm.
class RobotSupervisor
{
public:
  using RunSequence = openarm_messages::action::RunSequence;
  using FsmCommand = openarm_messages::srv::FsmCommand;
  using FsmState = openarm_messages::msg::FsmState;
  using GoalHandle = rclcpp_action::ServerGoalHandle<RunSequence>;

  /// Builds the owned SequenceFsm; does not advertise anything yet (see start()).
  RobotSupervisor(rclcpp::Node::SharedPtr node, std::shared_ptr<SequenceSource> source,
                  std::shared_ptr<ControlModeProbe> mode_probe,
                  std::shared_ptr<BuiltinActionRegistry> builtins);

  // Advertises everything and publishes the first state. Call after the
  // controllers are up.
  void start();

  // Launch-time convenience: run one sequence immediately, the way this node
  // behaved before it became a server. Empty name = just sit in IDLE.
  void autostart(const std::string& sequence_name);

private:
  /// RunSequence goal callback: rejects an empty sequence_name, ESTOP/FAULT
  /// (clear_fault first), TEACHING (exit teach first), or anything already
  /// running (one goal at a time - see file header comment); otherwise
  /// accepts and executes.
  rclcpp_action::GoalResponse handleGoal(const rclcpp_action::GoalUUID& uuid,
                                         std::shared_ptr<const RunSequence::Goal> goal);
  /// RunSequence cancel callback: forwards to the running SequenceFsm's cancel().
  rclcpp_action::CancelResponse handleCancel(const std::shared_ptr<GoalHandle>& goal_handle);
  /// RunSequence accepted callback: stores `goal_handle` as active_goal_ and starts the FSM.
  void handleAccepted(const std::shared_ptr<GoalHandle>& goal_handle);

  /// FsmCommand service callback: dispatches pause/resume/step/cancel/estop/
  /// clear_fault/enter_teach/exit_teach.
  void handleCommand(const std::shared_ptr<FsmCommand::Request> request,
                     std::shared_ptr<FsmCommand::Response> response);

  /// SequenceFsm transition callback: mirrors progress onto the active goal's
  /// feedback and publishes state.
  void onSequenceTransition(const SequenceProgress& progress);
  /// SequenceFsm finished callback: reports the result on active_goal_,
  /// transitions robot_state_ (IDLE or FAULT), and publishes state.
  void onSequenceFinished(bool success, const std::string& error_message, int steps_completed);

  /// Updates robot_state_ and publishes the new state.
  void setRobotState(RobotState state);
  /// Publishes buildStateMessage() on state_pub_.
  void publishState();
  /// Assembles the current FsmState message from robot_state_/fault_reason_/last_progress_.
  FsmState buildStateMessage() const;

  /// Enters TEACHING mode (only valid in torque control mode); `message` explains a refusal.
  bool enterTeach(std::string& message);
  /// Leaves TEACHING mode back to IDLE.
  bool exitTeach(std::string& message);

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<SequenceSource> source_;
  std::shared_ptr<ControlModeProbe> mode_probe_;
  std::shared_ptr<BuiltinActionRegistry> builtins_;
  std::unique_ptr<SequenceFsm> fsm_;
  rclcpp::Logger logger_;

  rclcpp_action::Server<RunSequence>::SharedPtr run_server_;
  rclcpp::Service<FsmCommand>::SharedPtr command_service_;
  rclcpp::Publisher<FsmState>::SharedPtr state_pub_;

  RobotState robot_state_ = RobotState::BOOTING;
  std::string fault_reason_;

  // Held while a sequence runs so transitions can be mirrored onto the goal as
  // feedback and the result can be reported when it ends.
  std::shared_ptr<GoalHandle> active_goal_;
  SequenceProgress last_progress_;
};

}  // namespace sequence_executor
