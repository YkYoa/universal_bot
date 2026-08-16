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

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <trajectory_shapes/trajectory_shapes.hpp>

namespace qvic_2026 {

using sequence_executor::BuiltinAction;
using sequence_executor::BuiltinActionRegistry;
using sequence_executor::BuiltinContext;

namespace {

constexpr const char* kMotion = sequence_executor::kModeMotion;

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
BuiltinAction homeBothArms()
{
  BuiltinAction action;
  action.id = "action_01";
  action.label = "Home both arms";
  action.description = "Move both arms to homePoses, then hold the hands at their home posture.";
  action.required_control_mode = kMotion;

  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    auto skill = ctx.skill;
    auto hand = ctx.hand;
    auto cancelled = ctx.cancelled;
    auto logger = ctx.node->get_logger();

    // Named pose, not the DB's raw 7-per-arm waypoint vector: "both_arms"
    // has a <group_state name="home"> in openarm_bimanual.srdf, resolved
    // server-side joint-by-name (MoveToNamedPoseSkill), so it stays correct
    // even though "both_arms" is 16-DOF (not 14) under ee_type:=amazing_hand
    // - see amazing_hand_connector's joint comment in openarm_robot.xacro.
    // moveToJoint's raw vector has no such flexibility: robot_skills_node
    // aborts outright if the vector's length doesn't exactly match the
    // live group's DOF.
    skill->moveToNamedPose(
      "both_arms", "home", "safe_rrt", 0.0, 0.0,
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

// ── action_02: bimanual wave ─────────────────────────────────────────────────
namespace {

constexpr const char* kWaveProfile = "safe_rrt";

// The two arc sections, in the order the arms sweep them.
constexpr const char* kLeftArcSection = "waveEllipse";
constexpr const char* kRightArcSection = "waveEllipseR";
constexpr const char* kHomeSection = "homePoses";

// Everything read from the store before the first move, so a missing waypoint
// fails with the arms still stationary.
struct WaveData
{
  std::vector<double> home;        // 14: left 7 then right 7
  std::vector<double> ready;       // 14: the first point of each arc
  std::vector<double> out;         // interleaved arc, start -> end
  std::vector<double> back;        // the same, reversed
};

std::vector<double> interleave(const std::vector<std::vector<double>>& left,
                               const std::vector<std::vector<double>>& right,
                               bool reversed)
{
  std::vector<double> flat;
  flat.reserve(left.size() * 14);
  for (std::size_t i = 0; i < left.size(); ++i) {
    const std::size_t k = reversed ? left.size() - 1 - i : i;
    flat.insert(flat.end(), left[k].begin(), left[k].end());
    flat.insert(flat.end(), right[k].begin(), right[k].end());
  }
  return flat;
}

BuiltinAction waveBothArms()
{
  BuiltinAction action;
  action.id = "action_02";
  action.label = "Wave both arms";
  action.description =
    "Home, move to the start of the arc, then sweep the ellipse out and back "
    "with both arms moving together, until cancelled.";
  action.required_control_mode = kMotion;

  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    auto skill = ctx.skill;
    auto source = ctx.source;
    auto cancelled = ctx.cancelled;
    auto logger = ctx.node->get_logger();

    auto wave = std::make_shared<WaveData>();
    try {
      wave->home = source->loadWaypoint(std::string(kHomeSection) + "/laHomeAngle");
      const auto right_home = source->loadWaypoint(std::string(kHomeSection) + "/raHomeAngle");
      wave->home.insert(wave->home.end(), right_home.begin(), right_home.end());

      const auto left = source->loadSection(kLeftArcSection);
      const auto right = source->loadSection(kRightArcSection);
      if (left.size() != right.size()) {
        done(false, std::string(kLeftArcSection) + " has " + std::to_string(left.size()) +
                    " waypoints but " + kRightArcSection + " has " +
                    std::to_string(right.size()) +
                    " - the two arms cannot be interleaved unless they match");
        return;
      }
      if (left.empty()) {
        done(false, std::string(kLeftArcSection) + " is empty");
        return;
      }

      wave->ready = left.front();
      wave->ready.insert(wave->ready.end(), right.front().begin(), right.front().end());
      wave->out = interleave(left, right, false);
      wave->back = interleave(left, right, true);
    } catch (const std::exception& e) {
      done(false, std::string("could not read the wave data: ") + e.what() +
                  " (seed the store: ros2 run qvic_2026 sequence_store_cli.py import "
                  "--file <pkg>/config/sequence.yaml)");
      return;
    }

    auto sweep = std::make_shared<std::function<void(bool)>>();
    auto cycle = std::make_shared<int>(0);

    *sweep = [skill, wave, cancelled, logger, done, sweep, cycle](bool forward) {
      if (cancelled()) {
        done(false, "cancelled");
        return;
      }
      RCLCPP_INFO(logger, "action_02: sweep %d (%s)", ++(*cycle), forward ? "out" : "back");

      skill->moveToJointSequence(
        "both_arms", forward ? wave->out : wave->back, kWaveProfile, 0.0, 0.0,
        [sweep, forward, done](bool ok, const std::string& error) {
          if (!ok) {
            done(false, error);
            return;
          }
          (*sweep)(!forward);
        });
    };

    skill->moveToJoint(
      "both_arms", wave->home, kWaveProfile, 0.0, 0.0,
      [skill, wave, cancelled, logger, done, sweep](bool ok, const std::string& error) {
        if (!ok) {
          done(false, "home move failed: " + error);
          return;
        }
        if (cancelled()) {
          done(false, "cancelled");
          return;
        }
        RCLCPP_INFO(logger, "action_02: homed, moving to the start of the arc");

        skill->moveToJoint(
          "both_arms", wave->ready, kWaveProfile, 0.0, 0.0,
          [sweep, cancelled, done](bool ready_ok, const std::string& ready_error) {
            if (!ready_ok) {
              done(false, "could not reach the start of the arc: " + ready_error);
              return;
            }
            if (cancelled()) {
              done(false, "cancelled");
              return;
            }
            (*sweep)(true);
          });
      });
  };

  return action;
}

// BuiltinAction action_03()
// {
//   BuiltinAction action;
//   action.id = "action_03";
//   action.label = "greeting";
//   action.description = "Greeting in 5-6 seconds";
//   action.required_control_mode = kModeMotion;
//   action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done){
//   auto skill = ctx.skill;
//   auto source = ctx.source;
//   auto cancelled = ctx.cancelled;
//   auto logger = ctx.node->get_logger();

//   auto 
//   };
// }

}  // namespace

}  // namespace

void registerQvicActions(BuiltinActionRegistry& registry)
{
  registry.add(homeBothArms());
  registry.add(waveBothArms());

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
