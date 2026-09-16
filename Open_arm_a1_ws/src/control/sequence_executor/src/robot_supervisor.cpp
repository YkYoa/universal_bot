#include "sequence_executor/robot_supervisor.hpp"

#include <utility>

#include "sequence_executor/hand_gripper_client.hpp"
#include "sequence_executor/scene_client.hpp"
#include "sequence_executor/skill_client.hpp"

namespace sequence_executor {

namespace {
// action_01 ("Home both arms") - the one existing builtin that already does
// a collision-checked, named-pose return to a known-safe pose for both
// arms. abort_to_home reuses it rather than defining a second, parallel
// notion of "home".
constexpr const char* kHomeSequenceName = "builtin:action_01";
}  // namespace

const char* toString(RobotState state)
{
  switch (state) {
    case RobotState::BOOTING:  return "BOOTING";
    case RobotState::IDLE:     return "IDLE";
    case RobotState::RUNNING:  return "RUNNING";
    case RobotState::PAUSED:   return "PAUSED";
    case RobotState::FAULT:    return "FAULT";
    case RobotState::ESTOP:    return "ESTOP";
    case RobotState::TEACHING: return "TEACHING";
  }
  return "UNKNOWN";
}

RobotSupervisor::RobotSupervisor(rclcpp::Node::SharedPtr node,
                                 std::shared_ptr<SequenceSource> source,
                                 std::shared_ptr<ControlModeProbe> mode_probe,
                                 std::shared_ptr<BuiltinActionRegistry> builtins)
  : node_(std::move(node)),
    source_(std::move(source)),
    mode_probe_(std::move(mode_probe)),
    builtins_(std::move(builtins)),
    logger_(node_->get_logger())
{
  SequenceFsm::Clients clients;
  clients.skill = std::make_shared<SkillClient>(node_);
  clients.hand = std::make_shared<HandGripperClient>(node_);
  clients.scene = std::make_shared<SceneClient>(node_);

  motor_enable_ = std::make_shared<MotorEnableClient>(node_);

  fsm_ = std::make_unique<SequenceFsm>(node_, source_, clients, mode_probe_, builtins_);
  fsm_->setCallbacks(
    [this](const SequenceProgress& progress) { onSequenceTransition(progress); },
    [this](bool success, const std::string& error, int steps) {
      onSequenceFinished(success, error, steps);
    });
}

void RobotSupervisor::start()
{
  using namespace std::placeholders;

  // transient_local: whoever subscribes next - the web page, the Android app,
  // a fresh `ros2 topic echo` - gets the current state on connect rather than
  // a blank panel until something happens to move.
  rclcpp::QoS qos(rclcpp::KeepLast(1));
  qos.transient_local().reliable();
  state_pub_ = node_->create_publisher<FsmState>("~/state", qos);

  run_server_ = rclcpp_action::create_server<RunSequence>(
    node_, "~/run_sequence",
    std::bind(&RobotSupervisor::handleGoal, this, _1, _2),
    std::bind(&RobotSupervisor::handleCancel, this, _1),
    std::bind(&RobotSupervisor::handleAccepted, this, _1));

  command_service_ = node_->create_service<FsmCommand>(
    "~/fsm_command",
    [this](const std::shared_ptr<FsmCommand::Request> request,
           std::shared_ptr<FsmCommand::Response> response) { handleCommand(request, response); });

  setRobotState(RobotState::IDLE);
  RCLCPP_INFO(logger_, "Robot supervisor ready. Sequences from %s", source_->describe().c_str());
}

void RobotSupervisor::autostart(const std::string& sequence_name)
{
  if (sequence_name.empty()) {
    return;
  }
  RCLCPP_INFO(logger_, "Autostarting '%s'", sequence_name.c_str());
  setRobotState(RobotState::RUNNING);
  fsm_->start(sequence_name, 0, 0.0, false);
}

// ── RunSequence action ──────────────────────────────────────────────────────

rclcpp_action::GoalResponse RobotSupervisor::handleGoal(
  const rclcpp_action::GoalUUID& /*uuid*/, std::shared_ptr<const RunSequence::Goal> goal)
{
  if (goal->sequence_name.empty()) {
    RCLCPP_WARN(logger_, "Rejecting goal: sequence_name is empty");
    return rclcpp_action::GoalResponse::REJECT;
  }
  if (robot_state_ == RobotState::ESTOP || robot_state_ == RobotState::FAULT) {
    RCLCPP_WARN(logger_, "Rejecting '%s': robot is in %s, clear_fault first",
                goal->sequence_name.c_str(), toString(robot_state_));
    return rclcpp_action::GoalResponse::REJECT;
  }
  if (robot_state_ == RobotState::TEACHING) {
    RCLCPP_WARN(logger_, "Rejecting '%s': exit teach mode first", goal->sequence_name.c_str());
    return rclcpp_action::GoalResponse::REJECT;
  }
  if (robot_state_ != RobotState::IDLE) {
    RCLCPP_WARN(logger_, "Rejecting '%s': '%s' is already running",
                goal->sequence_name.c_str(), last_progress_.sequence_name.c_str());
    return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse RobotSupervisor::handleCancel(
  const std::shared_ptr<GoalHandle>& /*goal_handle*/)
{
  std::string message;
  fsm_->cancel(message);
  RCLCPP_INFO(logger_, "Cancel requested: %s", message.c_str());
  return rclcpp_action::CancelResponse::ACCEPT;
}

void RobotSupervisor::handleAccepted(const std::shared_ptr<GoalHandle>& goal_handle)
{
  active_goal_ = goal_handle;
  const auto goal = goal_handle->get_goal();

  // Goal is already accepted from the caller's point of view - it will run,
  // just not necessarily this instant. Re-enable before starting it rather
  // than having rejected it back in handleGoal(): the caller (a sequence
  // picked from a menu, a builtin triggered by the FSM itself) has no
  // reason to know or care that the last E-stop left the hardware
  // components needing a fresh activate cycle - see motor_enable_client.hpp.
  if (!motors_enabled_) {
    std::string enable_message;
    RCLCPP_INFO(logger_, "'%s' accepted while motors are disabled - enabling first",
                goal->sequence_name.c_str());
    if (motor_enable_->enableAll(enable_message)) {
      motors_enabled_ = true;
      RCLCPP_INFO(logger_, "motors enabled: %s", enable_message.c_str());
    } else {
      RCLCPP_ERROR(logger_, "could not enable motors for '%s': %s",
                   goal->sequence_name.c_str(), enable_message.c_str());
      auto result = std::make_shared<RunSequence::Result>();
      result->success = false;
      result->error_message = "motors disabled and could not be re-enabled: " + enable_message;
      result->steps_completed = 0;
      active_goal_->abort(result);
      active_goal_.reset();
      fault_reason_ = result->error_message;
      setRobotState(RobotState::FAULT);
      return;
    }
  }

  setRobotState(RobotState::RUNNING);
  fault_reason_.clear();

  // Returns immediately - the FSM advances on ROS callbacks delivered by this
  // same executor, so there is no worker thread to join here.
  fsm_->start(goal->sequence_name, goal->repeat_override, goal->velocity_override,
              goal->dry_run);
}

// ── FsmCommand service ──────────────────────────────────────────────────────

void RobotSupervisor::handleCommand(const std::shared_ptr<FsmCommand::Request> request,
                                    std::shared_ptr<FsmCommand::Response> response)
{
  const std::string& command = request->command;
  std::string message;
  bool ok = false;

  if (command == "pause") {
    ok = fsm_->pause(message);
  } else if (command == "resume") {
    if (robot_state_ != RobotState::PAUSED) {
      message = "robot is not paused";
    } else if ((ok = fsm_->resume(message))) {
      setRobotState(RobotState::RUNNING);
    }
  } else if (command == "step") {
    if (robot_state_ != RobotState::PAUSED) {
      message = "single-stepping only works while paused";
    } else if ((ok = fsm_->singleStep(message))) {
      setRobotState(RobotState::RUNNING);
    }
  } else if (command == "cancel") {
    ok = fsm_->cancel(message);
  } else if (command == "estop") {
    // Unconditional, and it always succeeds: an e-stop that can be refused is
    // not an e-stop. motors_enabled_ goes false here too, not just
    // robot_state_ - clear_fault below only resets the latter, on the theory
    // that whatever the operator was worried about needs a real re-enable,
    // not just an acknowledgement, before anything moves again.
    fsm_->cancel(message);
    fault_reason_ = "emergency stop requested";
    setRobotState(RobotState::ESTOP);
    motors_enabled_ = false;
    ok = true;
    message = "stopped";
  } else if (command == "enable") {
    // Manual equivalent of the auto-enable handleAccepted() does before
    // starting a goal - exposed as its own command so a dashboard can offer
    // an explicit "Enable Motors" action instead of only finding out via a
    // goal's failure message.
    if (motors_enabled_) {
      ok = true;
      message = "motors already enabled";
    } else {
      ok = motor_enable_->enableAll(message);
      if (ok) {
        motors_enabled_ = true;
      }
    }
    publishState();
  } else if (command == "clear_fault") {
    if (robot_state_ != RobotState::FAULT && robot_state_ != RobotState::ESTOP) {
      message = "no fault to clear";
    } else {
      fault_reason_.clear();
      setRobotState(RobotState::IDLE);
      ok = true;
      message = "cleared";
    }
  } else if (command == "enter_teach") {
    ok = enterTeach(message);
  } else if (command == "exit_teach") {
    ok = exitTeach(message);
  } else if (command == "abort_to_home") {
    if (pending_home_after_abort_) {
      // Already on the way - a second click/retry shouldn't re-issue cancel
      // or queue a second home.
      ok = true;
      message = "already cancelling and homing";
    } else if (robot_state_ == RobotState::FAULT || robot_state_ == RobotState::ESTOP) {
      message = "clear the fault first (clear_fault), then abort_to_home";
    } else if (robot_state_ == RobotState::TEACHING) {
      message = "exit teach mode first (exit_teach), then abort_to_home";
    } else if (robot_state_ == RobotState::RUNNING || robot_state_ == RobotState::PAUSED) {
      // Set BEFORE calling cancel(), not after: cancel() finishes
      // synchronously (calls straight through to onSequenceFinished) when
      // there is no in-flight ExecuteSkill goal to wait on - e.g. a builtin
      // action that never touched clients_.skill, or a step between goals.
      // onSequenceFinished would see this flag still false if it were set
      // here after the call returns.
      pending_home_after_abort_ = true;
      std::string cancel_message;
      fsm_->cancel(cancel_message);
      ok = true;
      message = "cancelling - will home once stopped";
    } else {
      // IDLE (BOOTING never reaches handleCommand: the service isn't
      // advertised until start()). Nothing to cancel - home directly.
      startBuiltin(kHomeSequenceName);
      ok = true;
      message = "homing";
    }
  } else {
    message = "unknown command '" + command +
              "'; expected pause|resume|step|cancel|estop|clear_fault|enter_teach|exit_teach|"
              "abort_to_home|enable";
  }

  response->success = ok;
  response->message = message;
  RCLCPP_INFO(logger_, "command '%s': %s (%s)", command.c_str(), ok ? "ok" : "refused",
              message.c_str());
}

bool RobotSupervisor::enterTeach(std::string& message)
{
  if (robot_state_ != RobotState::IDLE) {
    message = "teach mode can only be entered from IDLE, robot is " +
              std::string(toString(robot_state_));
    return false;
  }
  const std::string mode = mode_probe_ ? mode_probe_->mode() : "unknown";
  if (mode != "torque") {
    // Refusing rather than pretending: without gravity compensation the arm
    // holds position, and an operator pulling on it is fighting the motors.
    message = "hand-guiding needs the arm in torque mode, it is in '" + mode +
              "'. Set control_mode: torque in hardware_config.yaml and restart "
              "the hardware.";
    return false;
  }
  setRobotState(RobotState::TEACHING);
  message = "teach mode active - gravity compensation is holding the arm";
  return true;
}

bool RobotSupervisor::exitTeach(std::string& message)
{
  if (robot_state_ != RobotState::TEACHING) {
    message = "not in teach mode";
    return false;
  }
  setRobotState(RobotState::IDLE);
  message = "teach mode ended";
  return true;
}

// ── SequenceFsm callbacks ───────────────────────────────────────────────────

void RobotSupervisor::onSequenceTransition(const SequenceProgress& progress)
{
  last_progress_ = progress;

  // The sequence layer pausing is what puts the robot layer into PAUSED - the
  // two are not independent, the robot state is a summary of the inner one.
  if (fsm_->isPaused() && robot_state_ == RobotState::RUNNING) {
    setRobotState(RobotState::PAUSED);
    return;
  }

  publishState();

  if (active_goal_ && active_goal_->is_executing()) {
    auto feedback = std::make_shared<RunSequence::Feedback>();
    feedback->state = buildStateMessage();
    active_goal_->publish_feedback(feedback);
  }
}

void RobotSupervisor::onSequenceFinished(bool success, const std::string& error_message,
                                         int steps_completed)
{
  auto result = std::make_shared<RunSequence::Result>();
  result->success = success;
  result->error_message = error_message;
  result->steps_completed = steps_completed;

  // A cancel is an operator decision, not a malfunction - back to IDLE, no
  // fault to clear, same as a run finishing on its own.
  const bool reached_idle = success || error_message == "cancelled";
  if (reached_idle) {
    setRobotState(RobotState::IDLE);
  } else {
    fault_reason_ = error_message;
    setRobotState(RobotState::FAULT);
  }

  // Only ever set by abort_to_home's own cancel - consumed here regardless
  // of whether this run happened to finish on its own right as the cancel
  // landed. Cleared even when NOT reached_idle: a real fault means home
  // stays a no-op rather than driving motion through/around it - clear_fault
  // is the deliberate step for that, not an implicit side effect here.
  const bool should_home = pending_home_after_abort_ && reached_idle;
  pending_home_after_abort_ = false;

  if (active_goal_) {
    if (success) {
      active_goal_->succeed(result);
    } else if (error_message == "cancelled" && active_goal_->is_canceling()) {
      active_goal_->canceled(result);
    } else {
      active_goal_->abort(result);
    }
    active_goal_.reset();
  }

  if (should_home) {
    startBuiltin(kHomeSequenceName);
  }
}

void RobotSupervisor::startBuiltin(const std::string& sequence_name)
{
  setRobotState(RobotState::RUNNING);
  fault_reason_.clear();
  fsm_->start(sequence_name, 0, 0.0, false);
}

// ── state plumbing ──────────────────────────────────────────────────────────

void RobotSupervisor::setRobotState(RobotState state)
{
  if (robot_state_ != state) {
    RCLCPP_INFO(logger_, "robot: %s -> %s", toString(robot_state_), toString(state));
  }
  robot_state_ = state;
  publishState();
}

RobotSupervisor::FsmState RobotSupervisor::buildStateMessage() const
{
  FsmState msg;
  msg.stamp = node_->now();
  msg.robot_state = toString(robot_state_);
  msg.sequence_name = last_progress_.sequence_name;
  msg.sequence_state =
    last_progress_.state == SeqState::IDLE ? "" : toString(last_progress_.state);
  msg.step_index = last_progress_.step_index;
  msg.step_total = last_progress_.step_total;
  msg.step_name = last_progress_.step_name;
  msg.step_type = last_progress_.step_type;
  msg.loop_index = last_progress_.loop_index;
  msg.loop_total = last_progress_.loop_total;
  msg.control_mode_active = mode_probe_ ? mode_probe_->mode() : "unknown";
  msg.progress = last_progress_.progress;
  msg.fault_reason = fault_reason_.empty() ? last_progress_.fault_reason : fault_reason_;
  msg.motors_enabled = motors_enabled_;
  return msg;
}

void RobotSupervisor::publishState()
{
  if (state_pub_) {
    state_pub_->publish(buildStateMessage());
  }
}

}  // namespace sequence_executor
