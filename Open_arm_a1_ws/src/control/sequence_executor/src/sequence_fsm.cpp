#include "sequence_executor/sequence_fsm.hpp"

#include <algorithm>
#include <chrono>

#include "sequence_executor/step_parser.hpp"

namespace sequence_executor {

namespace {

// The gripper controller's travel, ported from the old hand/gripper wiring.
constexpr double kGripperOpenPosition = 0.044;
constexpr double kGripperClosePosition = 0.0;
constexpr double kGripperDurationS = 1.0;

// How often a holding wait/teach_hold step checks its deadline. Coarse on
// purpose - it only has to be finer than a human notices in a hold.
constexpr std::chrono::milliseconds kWaitTickPeriod{50};

bool isSceneStep(const std::string& type)
{
  return type == "add_object" || type == "remove_object" || type == "attach_object" ||
         type == "detach_object" || type == "allow_collision" || type == "disallow_collision";
}

std::string sideToArm(const std::string& side)
{
  return side == "right" ? "right_arm" : "left_arm";
}

}  // namespace

const char* toString(SeqState state)
{
  switch (state) {
    case SeqState::IDLE:           return "IDLE";
    case SeqState::LOADING:        return "LOADING";
    case SeqState::VALIDATING:     return "VALIDATING";
    case SeqState::STEP_PLANNING:  return "STEP_PLANNING";
    case SeqState::STEP_EXECUTING: return "STEP_EXECUTING";
    case SeqState::STEP_DONE:      return "STEP_DONE";
    case SeqState::LOOP_CHECK:     return "LOOP_CHECK";
    case SeqState::COMPLETED:      return "COMPLETED";
    case SeqState::FAILED:         return "FAILED";
    case SeqState::CANCELLED:      return "CANCELLED";
  }
  return "UNKNOWN";
}

SequenceFsm::SequenceFsm(rclcpp::Node::SharedPtr node, std::shared_ptr<SequenceSource> source,
                         Clients clients, std::shared_ptr<ControlModeProbe> mode_probe,
                         std::shared_ptr<BuiltinActionRegistry> builtins)
  : node_(std::move(node)),
    source_(std::move(source)),
    clients_(std::move(clients)),
    mode_probe_(std::move(mode_probe)),
    builtins_(std::move(builtins)),
    logger_(node_->get_logger())
{
}

void SequenceFsm::setCallbacks(TransitionCallback on_transition, FinishedCallback on_finished)
{
  on_transition_ = std::move(on_transition);
  on_finished_ = std::move(on_finished);
}

bool SequenceFsm::isRunning() const
{
  switch (progress_.state) {
    case SeqState::LOADING:
    case SeqState::VALIDATING:
    case SeqState::STEP_PLANNING:
    case SeqState::STEP_EXECUTING:
    case SeqState::STEP_DONE:
    case SeqState::LOOP_CHECK:
      return true;
    default:
      return false;
  }
}

void SequenceFsm::transition(SeqState state)
{
  progress_.state = state;
  if (on_transition_) {
    on_transition_(progress_);
  }
}

void SequenceFsm::fail(const std::string& reason)
{
  progress_.fault_reason = reason;
  RCLCPP_ERROR(logger_, "[%s] FAILED: %s", progress_.sequence_name.c_str(), reason.c_str());
  transition(SeqState::FAILED);
  finish(false, reason);
}

void SequenceFsm::finish(bool success, const std::string& error_message)
{
  // Anything still in flight belongs to a run that no longer exists.
  ++run_id_;
  paused_ = false;
  pause_requested_ = false;
  single_step_ = false;
  cancel_requested_ = false;
  if (wait_timer_) {
    wait_timer_->cancel();
  }
  builtin_ = nullptr;
  if (on_finished_) {
    on_finished_(success, error_message, progress_.steps_completed);
  }
}

void SequenceFsm::start(const std::string& name, int repeat_override, double velocity_override,
                        bool dry_run)
{
  ++run_id_;
  progress_ = SequenceProgress{};
  progress_.sequence_name = name;
  paused_ = false;
  pause_requested_ = false;
  single_step_ = false;
  cancel_requested_ = false;
  velocity_override_ = velocity_override;
  dry_run_ = dry_run;
  builtin_ = nullptr;
  spec_ = SequenceSpec{};

  transition(SeqState::LOADING);

  const std::string builtin_id = BuiltinActionRegistry::idFromSequenceName(name);
  if (!builtin_id.empty()) {
    builtin_ = builtins_ ? builtins_->find(builtin_id) : nullptr;
    if (!builtin_) {
      fail("no builtin action '" + builtin_id + "' is registered");
      return;
    }
    spec_.name = name;
    spec_.required_control_mode = builtin_->required_control_mode;
    progress_.step_total = 1;
    progress_.loop_total = 1;
    remaining_loops_ = 1;
  } else {
    try {
      spec_ = source_->loadSequence(name);
    } catch (const std::exception& e) {
      fail(std::string("could not load sequence: ") + e.what());
      return;
    }
    progress_.step_total = static_cast<int>(std::count_if(
      spec_.steps.begin(), spec_.steps.end(), [](const Step& s) { return s.enabled; }));
    if (progress_.step_total == 0) {
      fail("sequence '" + name + "' has no enabled steps");
      return;
    }
    remaining_loops_ = repeat_override != 0 ? repeat_override : spec_.repeat;
    if (remaining_loops_ == 0) {
      remaining_loops_ = 1;
    }
    progress_.loop_total = remaining_loops_;
  }

  transition(SeqState::VALIDATING);
  const std::string problem = validate();
  if (!problem.empty()) {
    fail(problem);
    return;
  }

  RCLCPP_INFO(logger_, "[%s] starting: %d step(s), repeat %d%s", name.c_str(),
              progress_.step_total, remaining_loops_, dry_run_ ? ", dry run" : "");

  if (builtin_) {
    runBuiltin();
    return;
  }

  progress_.step_index = 0;
  runStep();
}

std::string SequenceFsm::validate()
{
  const std::string active_mode = mode_probe_ ? mode_probe_->mode() : "unknown";

  if (!modeIsCompatible(spec_.required_control_mode, active_mode)) {
    return "sequence needs control mode '" + spec_.required_control_mode +
           "' but the arm came up in '" + active_mode +
           "'. The mode is fixed at startup - change control_mode in "
           "hardware_config.yaml and restart the hardware to run this.";
  }

  if (builtin_) {
    return {};
  }

  for (const auto& step : spec_.steps) {
    if (!step.enabled) {
      continue;
    }
    const std::string where =
      "step " + std::to_string(step.index) + " (" + step.name + ")";

    if (!modeIsCompatible(step.required_control_mode, active_mode)) {
      return where + " needs control mode '" + step.required_control_mode +
             "' but the arm is in '" + active_mode + "'";
    }

    // Resolve every reference now, so a typo fails before anything moves.
    if (!step.waypoint.empty() && !source_->hasWaypoint(step.waypoint)) {
      return where + ": no waypoint '" + step.waypoint + "'";
    }
    if (!step.right_waypoint.empty() && !source_->hasWaypoint(step.right_waypoint)) {
      return where + ": no waypoint '" + step.right_waypoint + "'";
    }
    if (!step.section.empty() && step.type == "move_joint_sequence" &&
        !source_->hasSection(step.section)) {
      return where + ": no waypoint section '" + step.section + "'";
    }
    if (!step.right_section.empty() && !source_->hasSection(step.right_section)) {
      return where + ": no waypoint section '" + step.right_section + "'";
    }

    // Bimanual interleaving needs matching counts; discovering that mid-run
    // would abort halfway through a wave.
    if (step.type == "move_joint_sequence" && !step.right_section.empty()) {
      try {
        const auto left = source_->loadSection(step.section);
        const auto right = source_->loadSection(step.right_section);
        if (left.size() != right.size()) {
          return where + ": '" + step.section + "' has " + std::to_string(left.size()) +
                 " waypoints but '" + step.right_section + "' has " +
                 std::to_string(right.size()) + " - bimanual sections must match";
        }
      } catch (const std::exception& e) {
        return where + ": " + e.what();
      }
    }
  }
  return {};
}

void SequenceFsm::runStep()
{
  if (cancel_requested_) {
    transition(SeqState::CANCELLED);
    finish(false, "cancelled");
    return;
  }

  // Skip disabled steps without counting them against progress.
  while (progress_.step_index < static_cast<int>(steps().size()) &&
         !steps()[progress_.step_index].enabled) {
    ++progress_.step_index;
  }

  if (progress_.step_index >= static_cast<int>(steps().size())) {
    transition(SeqState::LOOP_CHECK);
    if (remaining_loops_ > 0) {
      --remaining_loops_;
    }
    if (remaining_loops_ == 0) {
      transition(SeqState::COMPLETED);
      finish(true, "");
      return;
    }
    ++progress_.loop_index;
    progress_.step_index = 0;
    runStep();
    return;
  }

  const Step& step = steps()[progress_.step_index];
  progress_.step_name = step.name;
  progress_.step_type = step.type;
  transition(SeqState::STEP_PLANNING);
  dispatch(step);
}

std::function<void(bool, const std::string&)> SequenceFsm::stepDone()
{
  const int run = run_id_;
  return [this, run](bool ok, const std::string& error) { onStepFinished(run, ok, error); };
}

void SequenceFsm::onStepFinished(int run, bool ok, const std::string& error_message)
{
  if (run != run_id_) {
    // A result from a run that has already ended - a hand goal landing after a
    // cancel, say. Driving the current run with it would advance somebody
    // else's step index.
    return;
  }

  if (!ok) {
    if (cancel_requested_) {
      transition(SeqState::CANCELLED);
      finish(false, "cancelled");
      return;
    }
    fail("step " + std::to_string(progress_.step_index) + " (" + progress_.step_name +
         ") failed: " + error_message);
    return;
  }

  ++progress_.steps_completed;
  if (progress_.step_total > 0) {
    const int total_steps = progress_.loop_total > 0
                              ? progress_.step_total * progress_.loop_total
                              : progress_.step_total;
    progress_.progress =
      std::min(1.0F, static_cast<float>(progress_.steps_completed) / static_cast<float>(total_steps));
  }
  transition(SeqState::STEP_DONE);
  advance();
}

void SequenceFsm::advance()
{
  ++progress_.step_index;

  if (cancel_requested_) {
    transition(SeqState::CANCELLED);
    finish(false, "cancelled");
    return;
  }

  // Pausing lands here, at a step boundary, with no trajectory in flight.
  if (pause_requested_ || single_step_) {
    pause_requested_ = false;
    single_step_ = false;
    paused_ = true;
    RCLCPP_INFO(logger_, "[%s] paused before step %d", progress_.sequence_name.c_str(),
                progress_.step_index);
    if (on_transition_) {
      on_transition_(progress_);
    }
    return;
  }

  runStep();
}

void SequenceFsm::dispatch(const Step& step)
{
  if (dry_run_) {
    RCLCPP_INFO(logger_, "[dry run] step %d: %s (%s)", step.index, step.name.c_str(),
                step.type.c_str());
    onStepFinished(run_id_, true, "");
    return;
  }

  if (step.type == "move_joint") {
    dispatchMoveJoint(step);
  } else if (step.type == "move_joint_sequence") {
    dispatchMoveJointSequence(step);
  } else if (step.type == "hand_pose") {
    dispatchHandPose(step);
  } else if (step.type == "gripper") {
    dispatchGripper(step);
  } else if (step.type == "wait" || step.type == "teach_hold") {
    dispatchWait(step);
  } else if (isSceneStep(step.type)) {
    dispatchScene(step);
  } else if (step.type == "set_speed") {
    // Purely a change to the running scaling, applied by velocityFor().
    spec_.velocity = step.velocity;
    spec_.acceleration = step.acceleration;
    onStepFinished(run_id_, true, "");
  } else {
    // move_pose and named_pose are defined in the step catalog but have no
    // dispatch yet; failing loudly beats moving to the wrong place.
    onStepFinished(run_id_, false,
                   "step type '" + step.type + "' is not implemented by the executor");
  }
}

void SequenceFsm::dispatchMoveJoint(const Step& step)
{
  std::vector<double> targets;
  try {
    if (!step.positions.empty()) {
      targets = step.positions;
    } else {
      targets = source_->loadWaypoint(step.waypoint);
      if (!step.right_waypoint.empty()) {
        const auto right = source_->loadWaypoint(step.right_waypoint);
        targets.insert(targets.end(), right.begin(), right.end());
      }
    }
  } catch (const std::exception& e) {
    onStepFinished(run_id_, false, e.what());
    return;
  }

  transition(SeqState::STEP_EXECUTING);
  clients_.skill->moveToJoint(
    step.arm, targets, spec_.planner_profile, velocityFor(step), accelerationFor(step),
    stepDone());
}

void SequenceFsm::dispatchMoveJointSequence(const Step& step)
{
  std::vector<std::vector<double>> left;
  std::vector<std::vector<double>> right;
  try {
    left = source_->loadSection(step.section);
    if (!step.right_section.empty()) {
      right = source_->loadSection(step.right_section);
    }
  } catch (const std::exception& e) {
    onStepFinished(run_id_, false, e.what());
    return;
  }

  // exclude_points is 1-indexed against the section's waypoint order - the
  // workaround for a point known to self-collide (see check_bimanual_collision).
  auto excluded = [&step](std::size_t zero_based) {
    const int one_based = static_cast<int>(zero_based) + 1;
    return std::find(step.exclude_points.begin(), step.exclude_points.end(), one_based) !=
           step.exclude_points.end();
  };

  std::vector<double> flat;
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (excluded(i)) {
      continue;
    }
    flat.insert(flat.end(), left[i].begin(), left[i].end());
    if (!right.empty()) {
      flat.insert(flat.end(), right[i].begin(), right[i].end());
    }
  }

  if (flat.empty()) {
    onStepFinished(run_id_, false, "every waypoint in '" + step.section + "' was excluded");
    return;
  }

  transition(SeqState::STEP_EXECUTING);
  clients_.skill->moveToJointSequence(
    step.arm, flat, spec_.planner_profile, velocityFor(step), accelerationFor(step),
    stepDone());
}

void SequenceFsm::dispatchHandPose(const Step& step)
{
  struct Goal
  {
    std::string arm;
    const std::vector<double>* values;
    bool is_yaw;
  };
  const std::vector<Goal> goals = {
    {"left_arm", &step.left_yaw, true},
    {"left_arm", &step.left_flex, false},
    {"right_arm", &step.right_yaw, true},
    {"right_arm", &step.right_flex, false},
  };

  pending_hand_goals_ = 0;
  for (const auto& goal : goals) {
    if (!goal.values->empty()) {
      ++pending_hand_goals_;
    }
  }
  if (pending_hand_goals_ == 0) {
    onStepFinished(run_id_, true, "");
    return;
  }

  hand_failed_ = false;
  hand_error_.clear();
  transition(SeqState::STEP_EXECUTING);

  // Yaw and flex, left and right, all fire together so one step is one
  // simultaneous posture rather than four sequential twitches. The counter
  // makes sure the step completes exactly once, after the last one lands;
  // stepDone() then drops the whole thing if the run ended in the meantime.
  auto done = stepDone();
  auto onOne = [this, done](bool ok, const std::string& error) {
    if (!ok && !hand_failed_) {
      hand_failed_ = true;
      hand_error_ = error;
    }
    if (--pending_hand_goals_ == 0) {
      done(!hand_failed_, hand_error_);
    }
  };

  for (const auto& goal : goals) {
    if (goal.values->empty()) {
      continue;
    }
    if (goal.is_yaw) {
      clients_.hand->setHandYaw(goal.arm, *goal.values, step.duration, onOne);
    } else {
      clients_.hand->setHandFlex(goal.arm, *goal.values, step.duration, onOne);
    }
  }
}

void SequenceFsm::dispatchGripper(const Step& step)
{
  transition(SeqState::STEP_EXECUTING);
  auto done = stepDone();
  const std::string arm = sideToArm(step.side);

  if (step.action == "open") {
    clients_.hand->openGripper(arm, kGripperOpenPosition, kGripperDurationS, done);
  } else {
    clients_.hand->closeGripper(arm, kGripperClosePosition, kGripperDurationS, done);
  }
}

void SequenceFsm::dispatchWait(const Step& step)
{
  transition(SeqState::STEP_EXECUTING);

  if (step.seconds <= 0.0) {
    onStepFinished(run_id_, true, "");
    return;
  }

  // A timer rather than a sleep: this thread is the executor, and blocking it
  // would stop the very callbacks a cancel needs to arrive on. The timer is
  // shared and long-lived (see the header) - here it is just re-armed.
  wait_deadline_ = node_->now() + rclcpp::Duration::from_seconds(step.seconds);
  wait_run_ = run_id_;
  if (!wait_timer_) {
    wait_timer_ = node_->create_wall_timer(kWaitTickPeriod, [this]() { onWaitTick(); });
  }
  wait_timer_->reset();
}

void SequenceFsm::onWaitTick()
{
  if (node_->now() < wait_deadline_) {
    return;
  }
  // Cancelling from inside the callback is safe; destroying would not be.
  wait_timer_->cancel();
  onStepFinished(wait_run_, true, "");
}

void SequenceFsm::dispatchScene(const Step& step)
{
  transition(SeqState::STEP_EXECUTING);
  clients_.scene->sendForStep(step, stepDone());
}

void SequenceFsm::runBuiltin()
{
  progress_.step_index = 0;
  progress_.step_name = builtin_->label;
  progress_.step_type = "builtin";
  transition(SeqState::STEP_EXECUTING);

  if (dry_run_) {
    RCLCPP_INFO(logger_, "[dry run] builtin '%s'", builtin_->id.c_str());
    onStepFinished(run_id_, true, "");
    return;
  }

  BuiltinContext context;
  context.node = node_;
  context.skill = clients_.skill;
  context.hand = clients_.hand;
  context.scene = clients_.scene;
  context.source = source_;
  context.cancelled = [this]() { return cancel_requested_; };

  builtin_->run(context, [this](bool ok, const std::string& error) {
    if (!ok) {
      if (cancel_requested_) {
        transition(SeqState::CANCELLED);
        finish(false, "cancelled");
        return;
      }
      fail("builtin action failed: " + error);
      return;
    }
    ++progress_.steps_completed;
    progress_.progress = 1.0F;
    transition(SeqState::COMPLETED);
    finish(true, "");
  });
}

double SequenceFsm::velocityFor(const Step& step) const
{
  if (velocity_override_ > 0.0) {
    return velocity_override_;
  }
  if (step.velocity > 0.0) {
    return step.velocity;
  }
  return spec_.velocity;   // 0 = let the planner profile decide
}

double SequenceFsm::accelerationFor(const Step& step) const
{
  if (step.acceleration > 0.0) {
    return step.acceleration;
  }
  return spec_.acceleration;
}

bool SequenceFsm::pause(std::string& message)
{
  if (paused_) {
    message = "already paused";
    return false;
  }
  if (!isRunning()) {
    message = "nothing is running";
    return false;
  }
  pause_requested_ = true;
  message = "will pause at the next step boundary";
  return true;
}

bool SequenceFsm::resume(std::string& message)
{
  if (!paused_) {
    message = "not paused";
    return false;
  }
  paused_ = false;
  message = "resumed";
  runStep();
  return true;
}

bool SequenceFsm::singleStep(std::string& message)
{
  if (!paused_) {
    message = "single-stepping only works while paused";
    return false;
  }
  paused_ = false;
  single_step_ = true;
  message = "running one step";
  runStep();
  return true;
}

bool SequenceFsm::cancel(std::string& message)
{
  if (!isRunning() && !paused_) {
    message = "nothing is running";
    return false;
  }
  cancel_requested_ = true;
  if (wait_timer_) {
    wait_timer_->cancel();
  }

  // Stops the trajectory now: robot_skills' handle_cancel calls
  // TrajectoryExecutionManager::stopExecution. The goal's result callback then
  // arrives with CANCELED and ends the run through the normal path.
  const bool goal_in_flight = !paused_ && clients_.skill->cancelActiveGoal();

  if (!goal_in_flight) {
    // Nothing to wait for: either we are paused between steps, or the step in
    // flight is not an ExecuteSkill goal (a hand pose, a wait timer), or the
    // skill server died before it ever acknowledged the goal. Ending the run
    // here is what makes cancel unconditional - a cancel that reports success
    // and leaves the machine stuck is worse than one that refuses. run_id_
    // moves on inside finish(), so any late callback is ignored.
    paused_ = false;
    transition(SeqState::CANCELLED);
    finish(false, "cancelled");
    message = "cancelled";
    return true;
  }

  message = "cancelling";
  return true;
}

}  // namespace sequence_executor
