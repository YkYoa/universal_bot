// -----------------------------------------------------------------------------
// qvic_actions.cpp
//
// Ten hardcoded action slots. Nine are empty; action_01 is filled in as a
// worked example of the shape the others should take.
//
// ── How to write one ─────────────────────────────────────────────────────────
//
// The body is asynchronous. It gets a BuiltinContext full of clients and a
// `done` callback, and it must return immediately - the thread it runs on is
// the same executor that will deliver its own action results, so blocking it
// deadlocks the node. Chain work through result callbacks, exactly the way
// SequenceFsm does:
//
//     action.run = [](BuiltinContext& ctx, DoneCallback done) {
//       ctx.skill->moveToJoint("left_arm", targets, "safe_rrt", 0, 0,
//         [&ctx, done](bool ok, const std::string& error) {
//           if (!ok) { done(false, error); return; }
//           if (ctx.cancelled()) { done(false, "cancelled"); return; }
//           ... next hop ...
//         });
//     };
//
// Three rules:
//   1. Call `done` exactly once, on every path. Forgetting it on an error
//      branch leaves the FSM waiting forever - that was the bug in the
//      callback chain this whole state machine replaced.
//   2. Check ctx.cancelled() between hops. An action that ignores it cannot
//      be stopped early, and the operator's cancel button will appear broken.
//   3. Set required_control_mode honestly. It is checked before the body runs,
//      against the mode the arm actually came up in.
//
// Waypoints come from the same store the sequences use, via ctx.source, so an
// action can reuse anything recorded with waypoint_recorder:
//
//     auto targets = ctx.source->loadWaypoint("homePoses/laHomeAngle");
//
// Capture by value, not reference: `ctx` is a stack local in SequenceFsm and
// is gone by the time an async callback fires. Copy out what you need first.
// -----------------------------------------------------------------------------
#include "qvic_2026/qvic_actions.hpp"

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace qvic_2026 {

using sequence_executor::BuiltinAction;
using sequence_executor::BuiltinActionRegistry;
using sequence_executor::BuiltinContext;

namespace {

// Everything that moves an arm needs the hardware in position or mit mode;
// torque mode ignores position commands outright.
constexpr const char* kMotion = sequence_executor::kModeMotion;

// A slot nobody has filled in yet. It fails rather than silently succeeding,
// so an empty action wired to a button on the Android app says so instead of
// looking like it worked.
BuiltinAction placeholder(const std::string& id, const std::string& label)
{
  BuiltinAction action;
  action.id = id;
  action.label = label;
  action.description = "Not implemented yet - fill in " + id + " in qvic_actions.cpp.";
  action.required_control_mode = sequence_executor::kModeAny;
  action.run = [id](BuiltinContext&, BuiltinAction::DoneCallback done) {
    done(false, "builtin action '" + id + "' has no body yet");
  };
  return action;
}

// ── action_01: worked example ────────────────────────────────────────────────
// Both arms to their home pose, then both hands to the home posture. Two async
// hops, showing the cancel check and the single-`done` discipline.
BuiltinAction homeBothArms()
{
  BuiltinAction action;
  action.id = "action_01";
  action.label = "Home both arms";
  action.description = "Move both arms to homePoses, then hold the hands at their home posture.";
  action.required_control_mode = kMotion;

  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    // Copy what the callbacks need - ctx does not outlive this call.
    auto skill = ctx.skill;
    auto hand = ctx.hand;
    auto source = ctx.source;
    auto cancelled = ctx.cancelled;
    auto logger = ctx.node->get_logger();

    std::vector<double> targets;
    try {
      targets = source->loadWaypoint("homePoses/laHomeAngle");
      const auto right = source->loadWaypoint("homePoses/raHomeAngle");
      targets.insert(targets.end(), right.begin(), right.end());
    } catch (const std::exception& e) {
      done(false, std::string("could not read the home waypoints: ") + e.what());
      return;
    }

    skill->moveToJoint(
      "both_arms", targets, "safe_rrt", 0.0, 0.0,
      [hand, cancelled, logger, done](bool ok, const std::string& error) {
        if (!ok) {
          done(false, "home move failed: " + error);
          return;
        }
        if (cancelled()) {
          done(false, "cancelled");
          return;
        }

        RCLCPP_INFO(logger, "action_01: arms home, closing hands to the home posture");

        // Both hands fire together; the counter makes sure `done` runs once.
        const std::vector<double> flex = {-0.0079, 0.0026, 1.2186, 1.2133};
        auto pending = std::make_shared<int>(2);
        auto failure = std::make_shared<std::string>();

        auto onOne = [pending, failure, done](bool hand_ok, const std::string& hand_error) {
          if (!hand_ok && failure->empty()) {
            *failure = hand_error;
          }
          if (--(*pending) == 0) {
            done(failure->empty(), *failure);
          }
        };

        hand->setHandFlex("left_arm", flex, 1.0, onOne);
        hand->setHandFlex("right_arm", flex, 1.0, onOne);
      });
  };

  return action;
}

}  // namespace

void registerQvicActions(BuiltinActionRegistry& registry)
{
  registry.add(homeBothArms());

  registry.add(placeholder("action_02", "Action 02"));
  registry.add(placeholder("action_03", "Action 03"));
  registry.add(placeholder("action_04", "Action 04"));
  registry.add(placeholder("action_05", "Action 05"));
  registry.add(placeholder("action_06", "Action 06"));
  registry.add(placeholder("action_07", "Action 07"));
  registry.add(placeholder("action_08", "Action 08"));
  registry.add(placeholder("action_09", "Action 09"));
  registry.add(placeholder("action_10", "Action 10"));
}

}  // namespace qvic_2026
