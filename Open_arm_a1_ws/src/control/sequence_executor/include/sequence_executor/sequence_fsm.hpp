#pragma once
// -----------------------------------------------------------------------------
// sequence_fsm.hpp
//
// The sequence-level state machine: walks one sequence's step list, once per
// loop iteration, reporting every transition.
//
// What changed from the interpreter this replaces:
//   - There is an actual state variable. The old version encoded its state
//     purely in which callback was in flight, so nothing could report where it
//     was, and nothing could interrupt it.
//   - Every failure funnels through one fail() call. The old version simply
//     stopped invoking continuations on error - runHome() failing left the
//     machine alive, idle, and silent.
//   - VALIDATING runs before any motion: every waypoint and section reference
//     is resolved and every step's control mode is checked against the
//     hardware's. A bad reference fails with the arm still stationary.
//   - Pause, resume, single-step, and cancel exist.
//
// Threading: none. The owning node runs a SingleThreadedExecutor, so action
// results, service handlers, and timers are all delivered on one thread and
// every member below is touched from that thread only. This is why there are
// no locks - adding a MultiThreadedExecutor would require adding them.
//
// Pause semantics: pausing takes effect at the next step boundary. A
// trajectory already in flight finishes. To stop immediately, cancel - that
// cancels the in-flight ExecuteSkill goal, which stops execution in
// robot_skills' handle_cancel.
// -----------------------------------------------------------------------------
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "sequence_executor/builtin_actions.hpp"
#include "sequence_executor/control_mode_probe.hpp"
#include "sequence_executor/hand_api_client.hpp"
#include "sequence_executor/hand_gripper_client.hpp"
#include "sequence_executor/scene_client.hpp"
#include "sequence_executor/sequence_source.hpp"
#include "sequence_executor/sequence_step.hpp"
#include "sequence_executor/skill_client.hpp"

namespace sequence_executor {

enum class SeqState
{
  IDLE,            // nothing loaded
  LOADING,         // fetching the sequence from its source
  VALIDATING,      // resolving references and checking control modes
  STEP_PLANNING,   // goal sent, waiting for the planner
  STEP_EXECUTING,  // trajectory running
  STEP_DONE,       // step finished, deciding what is next
  LOOP_CHECK,      // end of an iteration, deciding whether to repeat
  COMPLETED,
  FAILED,
  CANCELLED,
};

/// Human-readable name of `state`, for logs and the FsmState message.
const char* toString(SeqState state);

// A snapshot of everything the FsmState message carries about the sequence
// layer. Passed to the transition callback so the publisher stays a pure
// function of FSM state.
struct SequenceProgress
{
  SeqState state = SeqState::IDLE;
  std::string sequence_name;
  int step_index = -1;
  int step_total = 0;
  std::string step_name;
  std::string step_type;
  int loop_index = 0;
  int loop_total = 1;      // -1 = forever
  int steps_completed = 0;
  float progress = 0.0F;
  std::string fault_reason;
};

/// The sequence-level state machine - see the file header comment for the
/// full design rationale (single state variable, funnel-through-fail(),
/// pre-motion VALIDATING, pause/resume/step/cancel, single-threaded).
class SequenceFsm
{
public:
  using TransitionCallback = std::function<void(const SequenceProgress&)>;
  using FinishedCallback =
    std::function<void(bool success, const std::string& error_message, int steps_completed)>;

  /// The client set a sequence dispatches steps through.
  struct Clients
  {
    std::shared_ptr<SkillClient> skill;
    std::shared_ptr<HandGripperClient> hand;
    std::shared_ptr<SceneClient> scene;
    // Null when this robot has no REST-driven hand; hand_fingers steps then
    // fail validation instead of silently doing nothing.
    std::shared_ptr<HandApiClient> hand_api;
  };

  /// Wires up the source/clients/mode probe/builtins this FSM will dispatch
  /// through; does not start anything (see start()).
  SequenceFsm(rclcpp::Node::SharedPtr node, std::shared_ptr<SequenceSource> source,
              Clients clients, std::shared_ptr<ControlModeProbe> mode_probe,
              std::shared_ptr<BuiltinActionRegistry> builtins);

  /// Registers the callbacks invoked on every state transition and on run
  /// completion (success or failure).
  void setCallbacks(TransitionCallback on_transition, FinishedCallback on_finished);

  /// Kicks off `name`. Returns immediately; everything after this happens on
  /// ROS callbacks. `repeat_override` 0 means use the sequence's own repeat.
  /// `velocity_override` 0 means the same for speed. `dry_run` validates and
  /// walks the steps without sending a single motion goal.
  void start(const std::string& name, int repeat_override, double velocity_override, bool dry_run);

  // All no-ops when they do not apply, each returning why.
  /// Pauses at the next step boundary (see file header comment for why not
  /// immediately); `message` explains a no-op.
  bool pause(std::string& message);
  /// Resumes a paused run.
  bool resume(std::string& message);
  /// Runs exactly one more step then re-pauses; only valid while paused.
  bool singleStep(std::string& message);
  /// Cancels the run immediately, including any in-flight ExecuteSkill goal.
  bool cancel(std::string& message);

  /// True while a sequence is actively loaded/running (not IDLE/COMPLETED/FAILED/CANCELLED).
  bool isRunning() const;
  /// True if paused at a step boundary.
  bool isPaused() const { return paused_; }
  /// Current progress snapshot, as published in FsmState.
  const SequenceProgress& progress() const { return progress_; }

private:
  /// Sets progress_.state to `state` and invokes on_transition_.
  void transition(SeqState state);
  /// Records `reason` as the fault, transitions to FAILED, and finishes the run.
  void fail(const std::string& reason);
  /// Resets run state and invokes on_finished_ with the final outcome.
  void finish(bool success, const std::string& error_message);

  /// Resolves every reference and checks every control mode up front.
  /// Returns the reason it is unrunnable, or empty if it is fine.
  std::string validate();

  /// Dispatches the current step (or advances/finishes if none remain).
  void runStep();

  /// Every dispatch routes its result through here. `run` is the run_id_ the
  /// step was dispatched under: a callback arriving from an abandoned run - a
  /// hand goal that lands after a cancel, a result from the previous sequence -
  /// is dropped instead of driving the current one.
  void onStepFinished(int run, bool ok, const std::string& error_message);

  /// The result callback to hand to a client, tagged with the current run.
  std::function<void(bool, const std::string&)> stepDone();

  /// Moves to the next step (or the next loop iteration, or COMPLETED).
  void advance();

  // One dispatch branch per step type. Each ends by calling onStepFinished,
  // either directly or from a ROS callback.
  /// Routes `step` to the matching dispatchX() method by step.type.
  void dispatch(const Step& step);

  // Each mover comes in two forms. The `...Into` form takes the completion
  // callback, so move_groups can run several of them at once and join them;
  // the plain form is the same thing wired to this step's own completion.
  using Done = std::function<void(bool, const std::string&)>;
  /// Sends a single-joint-target (or named-pose/pose, per step fields) skill
  /// goal, wired to this step's own completion.
  void dispatchMoveJoint(const Step& step);
  /// Same as dispatchMoveJoint() but reports through `done` instead of the
  /// step's own completion - used when fanned out from dispatchMoveGroups().
  void dispatchMoveJointInto(const Step& step, Done done);
  /// Sends a joint-sequence (waypoint list) skill goal.
  void dispatchMoveJointSequence(const Step& step);
  /// Same as dispatchMoveJointSequence() but reports through `done`.
  void dispatchMoveJointSequenceInto(const Step& step, Done done);
  /// Sends the hand-pose goal(s) (whichever of left/right yaw/flex vectors
  /// are present, concurrently) to the hand client.
  void dispatchHandPose(const Step& step);
  /// Same as dispatchHandPose() but reports through `done`.
  void dispatchHandPoseInto(const Step& step, Done done);
  /// Sends an open/close gripper goal.
  void dispatchGripper(const Step& step);
  /// Sends a hand_fingers goal to hand_api_ (fails validation if hand_api is null).
  void dispatchHandFingers(const Step& step);

  // Every subsystem named in the step, fired together and joined once.
  /// Fans a move_groups step out into one dispatchXInto() call per subsystem
  /// with a target set, joining all of them into a single completion.
  void dispatchMoveGroups(const Step& step);
  /// Starts wait_timer_ for step.seconds (or holds indefinitely for teach_hold).
  void dispatchWait(const Step& step);
  /// Sends a scene edit (add/remove/attach/detach/allow/disallow/clear) via SceneClient.
  void dispatchScene(const Step& step);

  /// Runs the current step's builtin_ action (see BuiltinActionRegistry).
  void runBuiltin();

  // Sequence velocity, unless the step or the caller overrode it.
  /// Effective velocity for `step`: step override, else run-time
  /// velocity_override_, else the sequence's own default.
  double velocityFor(const Step& step) const;
  /// Effective acceleration for `step`, same precedence as velocityFor().
  double accelerationFor(const Step& step) const;

  // Enabled steps only - a disabled step is skipped without being counted.
  const std::vector<Step>& steps() const { return spec_.steps; }

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<SequenceSource> source_;
  Clients clients_;
  std::shared_ptr<ControlModeProbe> mode_probe_;
  std::shared_ptr<BuiltinActionRegistry> builtins_;
  rclcpp::Logger logger_;

  TransitionCallback on_transition_;
  FinishedCallback on_finished_;

  SequenceSpec spec_;
  SequenceProgress progress_;
  const BuiltinAction* builtin_ = nullptr;

  // Bumped on every start and every finish, so in-flight callbacks belonging
  // to a run that has already ended can be recognised and dropped.
  int run_id_ = 0;

  int remaining_loops_ = 0;      // -1 = forever
  double velocity_override_ = 0.0;
  bool dry_run_ = false;
  bool paused_ = false;
  bool pause_requested_ = false;
  bool single_step_ = false;
  bool cancel_requested_ = false;

  // One timer for the whole lifetime of the FSM, ticking only while a
  // wait/teach_hold step is holding. Created once and cancelled/restarted
  // rather than created per step: a per-step one-shot has to destroy itself
  // from inside its own callback, and if the next step is also a wait it
  // destroys the timer whose callback is still on the stack. That showed up as
  // an extra tick advancing the step index twice.
  rclcpp::TimerBase::SharedPtr wait_timer_;
  rclcpp::Time wait_deadline_;
  int wait_run_ = 0;
  /// wait_timer_ callback: checks wait_deadline_ and calls onStepFinished() once reached.
  void onWaitTick();

  // Counts the concurrent hand goals a hand_pose step fans out into.
  int pending_hand_goals_ = 0;
  bool hand_failed_ = false;
  std::string hand_error_;
};

}  // namespace sequence_executor
