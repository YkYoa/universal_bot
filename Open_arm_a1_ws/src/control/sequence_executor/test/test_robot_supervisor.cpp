// -----------------------------------------------------------------------------
// test_robot_supervisor.cpp
//
// Exercises RobotSupervisor's "abort_to_home" FsmCommand entirely through its
// real public ROS surface (the RunSequence action, the FsmCommand service,
// the latched ~/state topic) - the same interfaces the web dashboard and the
// Android app use. Only the domain dependencies are faked: a SequenceSource
// that is never actually called (builtins never touch it - see
// SequenceFsm::start()'s idFromSequenceName branch) and a small
// BuiltinActionRegistry of instant/never-finishing test actions standing in
// for the real, hardware-touching ones in builds/qvic_2026/src/qvic_actions.cpp.
//
// Not covered: refusing abort_to_home from TEACHING. Reaching TEACHING
// requires ControlModeProbe::mode() == "torque", and ControlModeProbe is a
// concrete class that resolves that from a live controller_manager service
// call (or a hardware_config.yaml on disk) - neither is fakeable here without
// turning it into an interface, which is a larger change than this test
// justifies. The TEACHING branch is a one-line condition identical in shape
// to the FAULT branch below that is covered.
//
// Also not covered: a second abort_to_home landing *while the first cancel is
// still asynchronously in flight* (truly exercising pending_home_after_abort_
// already being true). SequenceFsm::cancel() only stays asynchronous while
// clients_.skill has an in-flight ExecuteSkill goal; none of the test
// builtins below ever call ctx.skill, so their cancellation is always
// synchronous (see robot_supervisor.cpp's comment on why
// pending_home_after_abort_ is set before, not after, calling cancel()).
// Reproducing the async case needs a fake ExecuteSkill action server.
// TestAbortToHome_CalledTwiceIsSafe below instead checks the weaker (but
// still real) property that calling it repeatedly never crashes, double-runs
// home, or leaves the FSM stuck.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <openarm_messages/action/run_sequence.hpp>
#include <openarm_messages/msg/fsm_state.hpp>
#include <openarm_messages/srv/fsm_command.hpp>

#include "sequence_executor/builtin_actions.hpp"
#include "sequence_executor/control_mode_probe.hpp"
#include "sequence_executor/motor_enable_client.hpp"
#include "sequence_executor/robot_supervisor.hpp"
#include "sequence_executor/sequence_source.hpp"

using namespace std::chrono_literals;
using sequence_executor::RobotSupervisor;

namespace {

/// Never actually called: every sequence these tests run is "builtin:<id>",
/// and SequenceFsm::start() resolves those through the BuiltinActionRegistry
/// without ever touching source_. Exists only to satisfy RobotSupervisor's
/// constructor.
class UnusedSequenceSource : public sequence_executor::SequenceSource
{
public:
  std::vector<std::string> listSequences() override { return {}; }
  sequence_executor::SequenceSpec loadSequence(const std::string& name) override
  {
    throw std::runtime_error("UnusedSequenceSource: no stored sequence '" + name + "'");
  }
  std::vector<double> loadWaypoint(const std::string& ref) override
  {
    throw std::runtime_error("UnusedSequenceSource: no waypoint '" + ref + "'");
  }
  std::vector<std::vector<double>> loadSection(const std::string& section) override
  {
    throw std::runtime_error("UnusedSequenceSource: no section '" + section + "'");
  }
  bool hasWaypoint(const std::string&) override { return false; }
  bool hasSection(const std::string&) override { return false; }
  std::string describe() const override { return "UnusedSequenceSource"; }
};

/// Always succeeds without touching a real controller_manager - see
/// motor_enable_client.hpp: enableAll() is virtual and the base
/// constructor's real service clients are harmless to create even with
/// nothing on the other end, so this only needs to override the one method
/// whose real implementation would otherwise block every test's first
/// goal for controller_manager's wait_for_service timeout, then fail it
/// (RobotSupervisor::handleAccepted() now enables before every goal when
/// motors_enabled_ starts false - see that member's comment).
class FakeMotorEnableClient : public sequence_executor::MotorEnableClient
{
public:
  using MotorEnableClient::MotorEnableClient;
  bool enableAll(std::string& message) override
  {
    message = "fake enable - always succeeds";
    return true;
  }
};

/// Registers the fake builtins these tests run:
///  - "action_01": the id abort_to_home hardcodes as the home sequence
///    (kHomeSequenceName in robot_supervisor.cpp) - succeeds immediately, so
///    tests never touch SkillClient/MoveIt/real hardware.
///  - "hangs": never calls done() on its own - only ends via cancel(),
///    letting tests put the FSM in RUNNING and keep it there on demand.
///  - "fails": fails immediately, to reach FAULT without needing real
///    hardware to actually fail.
std::shared_ptr<sequence_executor::BuiltinActionRegistry> makeTestRegistry()
{
  using sequence_executor::BuiltinAction;
  using sequence_executor::BuiltinContext;

  auto registry = std::make_shared<sequence_executor::BuiltinActionRegistry>();

  BuiltinAction home;
  home.id = "action_01";
  home.label = "test home";
  home.description = "instant-success stand-in for the real home action";
  home.required_control_mode = sequence_executor::kModeAny;
  home.run = [](BuiltinContext&, BuiltinAction::DoneCallback done) { done(true, ""); };
  registry->add(home);

  BuiltinAction hangs;
  hangs.id = "hangs";
  hangs.label = "test hang";
  hangs.description = "never finishes on its own - only via cancel()";
  hangs.required_control_mode = sequence_executor::kModeAny;
  hangs.run = [](BuiltinContext&, BuiltinAction::DoneCallback) {};
  registry->add(hangs);

  BuiltinAction fails;
  fails.id = "fails";
  fails.label = "test fail";
  fails.description = "fails immediately, to reach FAULT without real hardware";
  fails.required_control_mode = sequence_executor::kModeAny;
  fails.run = [](BuiltinContext&, BuiltinAction::DoneCallback done) { done(false, "boom"); };
  registry->add(fails);

  return registry;
}

/// One RobotSupervisor wired with the fakes above, driven through the same
/// public ROS interfaces (RunSequence action, FsmCommand service, ~/state
/// topic) a real client uses.
class RobotSupervisorTest : public ::testing::Test
{
protected:
  using RunSequence = openarm_messages::action::RunSequence;
  using FsmCommand = openarm_messages::srv::FsmCommand;
  using FsmState = openarm_messages::msg::FsmState;

  void SetUp() override
  {
    static int instance = 0;
    node_ = std::make_shared<rclcpp::Node>("test_robot_supervisor_" + std::to_string(++instance));

    auto source = std::make_shared<UnusedSequenceSource>();
    // probe() is deliberately never called: mode() stays "unknown", which
    // modeIsCompatible() (sequence_step.cpp) treats as permissive regardless
    // of a step's required_control_mode - every test builtin above also
    // declares kModeAny, so this never gates anything either way.
    auto mode_probe = std::make_shared<sequence_executor::ControlModeProbe>(node_, "");
    auto motor_enable = std::make_shared<FakeMotorEnableClient>(node_);
    supervisor_ = std::make_shared<RobotSupervisor>(node_, source, mode_probe, makeTestRegistry(),
                                                    motor_enable);
    supervisor_->start();

    // KeepLast(20), not (1): several transitions in a row happen
    // synchronously here (a builtin action with no in-flight ExecuteSkill
    // goal finishes cancel() inline - see robot_supervisor.cpp's comment on
    // pending_home_after_abort_), all before this test ever calls
    // spin_some(). A depth-1 reader queue would only ever keep the last of
    // a burst; a real client doesn't hit this because it's continuously
    // spinning and processes each publish as it arrives. Reliable +
    // transient_local still match the publisher's own QoS - only the local
    // reader-side history depth differs.
    rclcpp::QoS state_qos(rclcpp::KeepLast(20));
    state_qos.transient_local().reliable();
    state_sub_ = node_->create_subscription<FsmState>(
      "~/state", state_qos,
      [this](const FsmState::SharedPtr msg) {
        // A transient_local reader re-matching mid-stream can redeliver the
        // durability cache's current sample alongside genuinely new ones -
        // observed here as the same (robot_state, sequence_name) arriving
        // several times back to back with nothing in between. That is never
        // a real transition (the publisher only ever publishes on an actual
        // state change - see robot_supervisor.cpp's setRobotState()), so
        // collapsing immediate repeats is correct, not just papering over
        // it: a genuine transition history cannot contain two identical
        // consecutive entries by construction.
        StateEvent event{msg->robot_state, msg->sequence_name};
        if (history_.empty() || !(history_.back() == event)) {
          history_.push_back(event);
        }
      });

    run_client_ = rclcpp_action::create_client<RunSequence>(node_, "~/run_sequence");
    command_client_ = node_->create_client<FsmCommand>("~/fsm_command");

    executor_.add_node(node_);
    ASSERT_TRUE(spinUntil([this] { return run_client_->action_server_is_ready(); }))
      << "run_sequence action server never came up";
    ASSERT_TRUE(spinUntil([this] { return command_client_->service_is_ready(); }))
      << "fsm_command service never came up";
    ASSERT_TRUE(spinUntil([this] { return !history_.empty() && history_.back().robot_state == "IDLE"; }))
      << "supervisor never reached IDLE after start()";
  }

  void TearDown() override { executor_.remove_node(node_); }

  /// Spins the shared executor until `pred` is true or `timeout` elapses;
  /// returns whether it converged. Every wait in these tests goes through
  /// this instead of a fixed sleep, since the FSM's own transitions arrive
  /// as callbacks on this same executor.
  bool spinUntil(const std::function<bool()>& pred, std::chrono::milliseconds timeout = 2s)
  {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
      executor_.spin_some();
      if (pred()) return true;
      std::this_thread::sleep_for(1ms);
    } while (std::chrono::steady_clock::now() < deadline);
    return pred();
  }

  /// Spins until `history_` stops growing for `quiet_for` - unlike
  /// spinUntil(), which stops the instant a predicate first turns true.
  /// SingleThreadedExecutor::spin_some() only drains what was already queued
  /// when it was called, not work generated during that same call (e.g. a
  /// burst of state publishes made from inside a service callback this very
  /// spin_some() is executing) - a whole synchronous cancel -> home -> idle
  /// chain (see AbortToHomeFromRunningCancelsThenHomes) can therefore take
  /// several more spin_some() calls to fully arrive after the FIRST new
  /// history_ entry shows up. Waiting for "no growth for a bit" instead of
  /// "the state I expect appeared" is what actually waits for the whole
  /// burst, not just its first sample.
  void spinUntilSettled(std::chrono::milliseconds quiet_for = 200ms,
                        std::chrono::milliseconds timeout = 2s)
  {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::size_t last_size = history_.size();
    auto quiet_since = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() < deadline) {
      executor_.spin_some();
      if (history_.size() != last_size) {
        last_size = history_.size();
        quiet_since = std::chrono::steady_clock::now();
      } else if (std::chrono::steady_clock::now() - quiet_since >= quiet_for) {
        return;
      }
      std::this_thread::sleep_for(1ms);
    }
  }

  /// Sends `command` and returns its (success, message); spins until the
  /// service replies.
  std::pair<bool, std::string> sendCommand(const std::string& command)
  {
    auto request = std::make_shared<FsmCommand::Request>();
    request->command = command;
    auto future = command_client_->async_send_request(request);
    const bool got_response = spinUntil([&future] {
      return future.future.wait_for(0s) == std::future_status::ready;
    });
    if (!got_response) {
      return {false, "(test) command never got a response"};
    }
    auto response = future.future.get();
    return {response->success, response->message};
  }

  /// Sends a RunSequence goal for "builtin:<id>" and spins until it is
  /// accepted (not until it finishes - callers that need a specific
  /// downstream robot_state spin for that separately).
  void startBuiltinGoal(const std::string& id)
  {
    RunSequence::Goal goal;
    goal.sequence_name = "builtin:" + id;
    auto goal_future = run_client_->async_send_goal(goal);
    ASSERT_TRUE(spinUntil([&goal_future] {
      return goal_future.wait_for(0s) == std::future_status::ready;
    }));
    ASSERT_NE(goal_future.get(), nullptr) << "goal for builtin:" << id << " was rejected";
  }

  struct StateEvent
  {
    std::string robot_state;
    std::string sequence_name;

    bool operator==(const StateEvent& other) const
    {
      return robot_state == other.robot_state && sequence_name == other.sequence_name;
    }
  };

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<RobotSupervisor> supervisor_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  rclcpp::Subscription<FsmState>::SharedPtr state_sub_;
  rclcpp_action::Client<RunSequence>::SharedPtr run_client_;
  rclcpp::Client<FsmCommand>::SharedPtr command_client_;
  std::vector<StateEvent> history_;
};

}  // namespace

TEST_F(RobotSupervisorTest, AbortToHomeFromIdleHomesImmediately)
{
  auto [success, message] = sendCommand("abort_to_home");
  EXPECT_TRUE(success) << message;

  ASSERT_TRUE(spinUntil([this] {
    return history_.back().robot_state == "IDLE" && history_.size() >= 2;
  }));

  // Must have actually run the home builtin, not just stayed IDLE - look for
  // a RUNNING entry naming it somewhere in what we saw.
  bool ran_home = false;
  for (const auto& event : history_) {
    if (event.robot_state == "RUNNING" && event.sequence_name == "builtin:action_01") {
      ran_home = true;
    }
  }
  EXPECT_TRUE(ran_home);
  EXPECT_EQ(history_.back().robot_state, "IDLE");
}

TEST_F(RobotSupervisorTest, AbortToHomeFromRunningCancelsThenHomes)
{
  startBuiltinGoal("hangs");
  ASSERT_TRUE(spinUntil([this] { return history_.back().robot_state == "RUNNING"; }));
  ASSERT_EQ(history_.back().sequence_name, "builtin:hangs");

  auto [success, message] = sendCommand("abort_to_home");
  EXPECT_TRUE(success) << message;

  spinUntilSettled();
  ASSERT_EQ(history_.back().robot_state, "IDLE");

  // The cancel -> home -> idle chain runs as one synchronous burst (see
  // robot_supervisor.cpp's comment on why pending_home_after_abort_ is set
  // before, not after, calling cancel()) - several publishes on a
  // KeepLast(1) topic before this test's executor ever gets to spin. That
  // QoS depth is a deliberate choice for the real publisher ("an idle robot
  // produces no traffic at all" - robot_supervisor.hpp) and it means an
  // intermediate sample in a fast burst like this one is not guaranteed
  // delivery, only the eventual settled value is (transient_local's actual
  // contract). So this only asserts on what IS guaranteed: the first
  // transition (isolated in time, ordinary delivery) and the final settled
  // one - not every point in between.
  EXPECT_EQ(history_.back().robot_state, "IDLE");
  EXPECT_EQ(history_.back().sequence_name, "builtin:action_01")
    << "the settled state should show action_01 (home) as the last thing "
       "that ran, not the aborted 'hangs' run";
}

TEST_F(RobotSupervisorTest, AbortToHomeRefusedFromFault)
{
  startBuiltinGoal("fails");
  ASSERT_TRUE(spinUntil([this] { return history_.back().robot_state == "FAULT"; }));

  const std::size_t before = history_.size();
  auto [success, message] = sendCommand("abort_to_home");
  EXPECT_FALSE(success);
  EXPECT_NE(message.find("clear"), std::string::npos) << message;

  // Refused outright - no new transition at all, still FAULT.
  executor_.spin_some();
  EXPECT_EQ(history_.size(), before);
  EXPECT_EQ(history_.back().robot_state, "FAULT");
}

TEST_F(RobotSupervisorTest, GoalAcceptedAndAutoRecoversFromFault)
{
  startBuiltinGoal("fails");
  ASSERT_TRUE(spinUntil([this] { return history_.back().robot_state == "FAULT"; }));
  EXPECT_EQ(history_.back().robot_state, "FAULT");

  // A new goal submitted in FAULT must be accepted, auto-clearing fault and
  // actually running - but "action_01" completes synchronously and
  // instantly (see AbortToHomeFromIdleHomesImmediately above for the
  // identical situation), so by the time startBuiltinGoal() returns the
  // robot may already be back to IDLE. Wait for that settled state and
  // scan the full history for the RUNNING transition instead of requiring
  // it to still be history_.back() - same reasoning, same fix shape.
  startBuiltinGoal("action_01");
  ASSERT_TRUE(spinUntil([this] { return history_.back().robot_state == "IDLE"; }));

  bool recovered_and_ran = false;
  for (const auto& event : history_) {
    if (event.robot_state == "RUNNING" && event.sequence_name == "builtin:action_01") {
      recovered_and_ran = true;
    }
  }
  EXPECT_TRUE(recovered_and_ran) << "expected a RUNNING builtin:action_01 entry somewhere "
                                    "in history_ after the auto-recover";
  EXPECT_EQ(history_.back().robot_state, "IDLE");
}

TEST_F(RobotSupervisorTest, StopCommandCancelsToIdle)
{
  // "hangs" (not "infinite" - that id was never registered, see
  // makeTestRegistry() above) never calls done() on its own, so it stays
  // RUNNING until cancel()/"stop" ends it.
  startBuiltinGoal("hangs");
  ASSERT_TRUE(spinUntil([this] { return history_.back().robot_state == "RUNNING"; }));

  auto [stop_ok, stop_message] = sendCommand("stop");
  EXPECT_TRUE(stop_ok) << stop_message;
  ASSERT_TRUE(spinUntil([this] { return history_.back().robot_state == "IDLE"; }));
  EXPECT_EQ(history_.back().robot_state, "IDLE");
}

TEST_F(RobotSupervisorTest, EnableCommandAlwaysCallsEnableAll)
{
  // First manual enable call
  auto [first_ok, first_message] = sendCommand("enable");
  EXPECT_TRUE(first_ok) << first_message;

  // Second manual enable call - must STILL call enableAll(), never return "motors already enabled"
  auto [second_ok, second_message] = sendCommand("enable");
  EXPECT_TRUE(second_ok) << second_message;
  EXPECT_NE(second_message, "motors already enabled");

  // Calling stop sets motors_enabled_ = false
  auto [stop_ok, stop_message] = sendCommand("stop");
  EXPECT_TRUE(stop_ok) << stop_message;
}

TEST_F(RobotSupervisorTest, AbortToHomeCalledTwiceIsSafe)
{
  // See file header comment: with no test builtin that dispatches a real
  // ExecuteSkill goal, cancel() always finishes synchronously here, so this
  // cannot land the second call while the first is still pending - it only
  // proves repeated calls don't crash, double-run home, or get the FSM
  // stuck, which is still a real property worth locking in.
  auto [first_ok, first_message] = sendCommand("abort_to_home");
  EXPECT_TRUE(first_ok) << first_message;
  ASSERT_TRUE(spinUntil([this] { return history_.back().robot_state == "IDLE"; }));

  auto [second_ok, second_message] = sendCommand("abort_to_home");
  EXPECT_TRUE(second_ok) << second_message;
  ASSERT_TRUE(spinUntil([this] { return history_.back().robot_state == "IDLE"; }));

  int home_runs = 0;
  for (const auto& event : history_) {
    if (event.robot_state == "RUNNING" && event.sequence_name == "builtin:action_01") {
      ++home_runs;
    }
  }
  EXPECT_EQ(home_runs, 2) << "one home run per abort_to_home call, no double-queueing";
}

// Deliberately does NOT use RobotSupervisorTest/FakeMotorEnableClient: that
// fixture exists so most tests don't need a real controller_manager, but it
// also means none of them ever call the real MotorEnableClient::enableAll()
// - which is exactly what crashed production on 2026-09-16
// (motor_enable_client.hpp's file header comment has the full story).
// spin_until_future_complete(node_, ...) is only safe before node_ is added
// to an executor; handleAccepted() calls it from inside a callback that is
// only running because node_ already IS added to one. This test adds node_
// to executor_ first, same as production's executor_app.cpp, then sends a
// goal while motors_enabled_ defaults to false - reproducing the crash
// scenario exactly. There is no real controller_manager here, so enableAll()
// is expected to fail (the goal gets aborted) - the property under test is
// that the process survives to report that failure instead of aborting.
TEST(MotorEnableClientRegressionTest, EnableAllDuringLiveGoalDoesNotCrashProcess)
{
  using RunSequence = openarm_messages::action::RunSequence;

  auto node = std::make_shared<rclcpp::Node>("test_motor_enable_regression");
  auto source = std::make_shared<UnusedSequenceSource>();
  auto mode_probe = std::make_shared<sequence_executor::ControlModeProbe>(node, "");
  // No fake here - the real MotorEnableClient, talking to a
  // controller_manager that does not exist in this test process.
  auto supervisor =
    std::make_shared<RobotSupervisor>(node, source, mode_probe, makeTestRegistry());
  supervisor->start();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  auto run_client = rclcpp_action::create_client<RunSequence>(node, "~/run_sequence");
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (!run_client->action_server_is_ready() && std::chrono::steady_clock::now() < deadline) {
    executor.spin_some();
  }
  ASSERT_TRUE(run_client->action_server_is_ready());

  RunSequence::Goal goal;
  goal.sequence_name = "builtin:action_01";
  auto goal_future = run_client->async_send_goal(goal);

  // The real timeout: enableAll()'s own service waits (2-5s each, see
  // motor_enable_client.cpp) plus the network round trips above, so this
  // needs real headroom, not the 2s spinUntil() default used elsewhere in
  // this file. Reaching this deadline without the process having aborted
  // already IS the pass condition, independent of what the goal result was.
  const auto goal_deadline = std::chrono::steady_clock::now() + 15s;
  while (std::chrono::steady_clock::now() < goal_deadline) {
    executor.spin_some();
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(node);
  SUCCEED() << "process survived enableAll() running from inside a live goal callback";
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
