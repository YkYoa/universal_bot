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
    // "safe_rrt" (OMPL), not "fast_ptp" (Pilz PTP): confirmed on real
    // hardware (2026-08-17) that Pilz PTP fails to plan for "both_arms" -
    // a composite group of two independent 7-DOF chains, not the single
    // chain Pilz's planners are built around - "Planning failed for group
    // both_arms". Single-arm actions don't hit this; only this multi-chain
    // named-pose target does.
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
        // Hand failure (e.g. ee_type:=none - no hand_j1/j2_controller
        // spawned at all, confirmed 2026-08-17: "left_hand_j2_controller/
        // follow_joint_trajectory not available") is logged but does NOT
        // fail the whole action - the arm-homing above is the part that
        // matters, and there's no ee_type visible here to skip this step
        // outright when no hand is attached.
        const std::vector<double> flex = {-0.0079, 0.0026, 1.2186, 1.2133};
        auto pending = std::make_shared<int>(2);
        auto failure = std::make_shared<std::string>();

        auto onOne = [pending, failure, done, logger](bool hand_ok, const std::string& hand_error) {
          if (!hand_ok && failure->empty()) {
            *failure = hand_error;
          }
          if (--(*pending) == 0) {
            if (!failure->empty()) {
              RCLCPP_WARN(logger, "action_01: hands not moved (%s) - arms are still home",
                          failure->c_str());
            }
            done(true, "");
          }
        };

        hand->setHandFlex("left_arm", flex, 1.0, onOne);
        hand->setHandFlex("right_arm", flex, 1.0, onOne);
      });
  };

  return action;
}

// ── action_02/action_08: single-arm ellipse wave ─────────────────────────
namespace {

// Pilz PTP (2026-08-17, was "safe_rrt"/OMPL-RRTConnect): PTP synthesizes a
// smooth trapezoidal velocity profile directly, instead of RRTConnect's
// sampling-based path + post-hoc TOTG retiming - noticeably less jerk/
// vibration for these simple point-to-point waypoints, none of which need
// RRTConnect's obstacle-avoidance capability anyway. A single moveToJoint
// call still stops at zero velocity at its end regardless of profile - see
// moveToJointSequence below for the multi-waypoint case, which already
// blends through interior waypoints instead of stopping at each one.
constexpr const char* kWaveProfile = "fast_ptp";

// The two arc sections, in the order the arms sweep them.
constexpr const char* kLeftArcSection = "waveEllipse";
constexpr const char* kRightArcSection = "waveEllipseR";
constexpr const char* kHomeSection = "homePoses";

// Everything read from the store before the first move, so a missing waypoint
// fails with the arm still stationary.
struct ArmArcData
{
  std::vector<double> home;   // 7
  std::vector<double> ready;  // 7: the first point of the arc
  std::vector<double> out;    // flattened arc, start -> end
  std::vector<double> back;   // the same, reversed
};

std::vector<double> flattenArc(const std::vector<std::vector<double>>& points, bool reversed)
{
  std::vector<double> flat;
  if (points.empty()) return flat;
  flat.reserve(points.size() * points.front().size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const std::size_t k = reversed ? points.size() - 1 - i : i;
    flat.insert(flat.end(), points[k].begin(), points[k].end());
  }
  return flat;
}

// Shared body for waveLeftArmEllipse()/waveRightArmEllipse() - same logic,
// only the arm name/section/home-waypoint/log-prefix differ.
void runArmArcWave(BuiltinContext& ctx, BuiltinAction::DoneCallback done,
                   const std::string& arm, const char* section,
                   const std::string& home_waypoint, const char* log_prefix)
{
  auto skill = ctx.skill;
  auto source = ctx.source;
  auto cancelled = ctx.cancelled;
  auto logger = ctx.node->get_logger();

  auto wave = std::make_shared<ArmArcData>();
  try {
    wave->home = source->loadWaypoint(home_waypoint);
    const auto points = source->loadSection(section);
    if (points.empty()) {
      done(false, std::string(section) + " is empty");
      return;
    }
    wave->ready = points.front();
    wave->out = flattenArc(points, false);
    wave->back = flattenArc(points, true);
  } catch (const std::exception& e) {
    done(false, std::string("could not read the wave data: ") + e.what() +
                " (seed the store: ros2 run qvic_2026 sequence_store_cli.py import "
                "--file <pkg>/config/sequence.yaml)");
    return;
  }

  auto sweep = std::make_shared<std::function<void(bool)>>();
  auto cycle = std::make_shared<int>(0);

  *sweep = [skill, wave, cancelled, logger, done, sweep, cycle, arm, log_prefix](bool forward) {
    if (cancelled()) {
      done(false, "cancelled");
      return;
    }
    RCLCPP_INFO(logger, "%s: sweep %d (%s)", log_prefix, ++(*cycle), forward ? "out" : "back");

    // moveToJointSequence, not repeated moveToJoint calls: this arc has 17+
    // interior waypoints, and MoveItCppPlannerManager's joint-sequence path
    // (blendJointSequenceCorners + TOTG + Ruckig) already rounds each
    // interior corner into one continuous blended motion instead of
    // stopping at every point - see its comments for how.
    skill->moveToJointSequence(
      arm, forward ? wave->out : wave->back, kWaveProfile, 0.0, 0.0,
      [sweep, forward, done](bool ok, const std::string& error) {
        if (!ok) {
          done(false, error);
          return;
        }
        (*sweep)(!forward);
      });
  };

  skill->moveToJoint(
    arm, wave->home, kWaveProfile, 0.0, 0.0,
    [skill, wave, cancelled, logger, done, sweep, arm, log_prefix](bool ok, const std::string& error) {
      if (!ok) {
        done(false, "home move failed: " + error);
        return;
      }
      if (cancelled()) {
        done(false, "cancelled");
        return;
      }
      RCLCPP_INFO(logger, "%s: homed, moving to the start of the arc", log_prefix);

      skill->moveToJoint(
        arm, wave->ready, kWaveProfile, 0.0, 0.0,
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
}

BuiltinAction waveLeftArmEllipse()
{
  BuiltinAction action;
  action.id = "action_02";
  action.label = "Wave left arm (ellipse)";
  action.description =
    "Home, move to the start of the ellipse arc, then sweep it out and back "
    "with the left arm, until cancelled. Split from the old bimanual "
    "action_02 (2026-08-17) - see action_08 for the right-arm counterpart.";
  action.required_control_mode = kMotion;
  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    runArmArcWave(ctx, done, "left_arm", kLeftArcSection,
                  std::string(kHomeSection) + "/laHomeAngle", "action_02");
  };
  return action;
}

BuiltinAction waveRightArmEllipse()
{
  BuiltinAction action;
  action.id = "action_08";
  action.label = "Wave right arm (ellipse)";
  action.description =
    "Home, move to the start of the ellipse arc, then sweep it out and back "
    "with the right arm, until cancelled. Right-arm counterpart of action_02, "
    "split from the old bimanual action_02 (2026-08-17).";
  action.required_control_mode = kMotion;
  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    runArmArcWave(ctx, done, "right_arm", kRightArcSection,
                  std::string(kHomeSection) + "/raHomeAngle", "action_08");
  };
  return action;
}

// ── action_03: head rotate ───────────────────────────────────────────────

constexpr double kHeadRotateDeg = 10.0;
constexpr double kHeadRotateRad = kHeadRotateDeg * M_PI / 180.0;
constexpr double kHeadRotateDurationSec = 0.84;

BuiltinAction headRotate()
{
  BuiltinAction action;
  action.id = "action_03";
  action.label = "Head rotate";
  action.description = "Sweep the head between -10 and +10 degrees, until cancelled.";
  // No arm involved - head_controller runs independently of the arm's
  // position/mit control mode.
  action.required_control_mode = sequence_executor::kModeAny;

  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    auto hand = ctx.hand;
    auto cancelled = ctx.cancelled;
    auto logger = ctx.node->get_logger();

    auto sweep = std::make_shared<std::function<void(bool)>>();
    auto cycle = std::make_shared<int>(0);

    *sweep = [hand, cancelled, logger, done, sweep, cycle](bool to_positive) {
      if (cancelled()) {
        done(false, "cancelled");
        return;
      }
      RCLCPP_INFO(logger, "action_03: sweep %d (%s%.0fdeg)", ++(*cycle),
                  to_positive ? "+" : "-", kHeadRotateDeg);

      hand->setHead(0.0, to_positive ? kHeadRotateRad : -kHeadRotateRad,
        kHeadRotateDurationSec,
        [sweep, to_positive, done](bool ok, const std::string& error) {
          if (!ok) {
            done(false, error);
            return;
          }
          (*sweep)(!to_positive);
        });
    };

    (*sweep)(true);
  };

  return action;
}

// ── action_04: wave left arm ─────────────────────────────────────────────
constexpr double kDeg = M_PI / 180.0;
const std::vector<double> kWaveLeftArmStart = {
  -40.0 * kDeg, -10.0 * kDeg, -20.0 * kDeg, 90.0 * kDeg, 88.0 * kDeg, 0.0, 0.0};
const std::vector<double> kWaveLeftArmEnd = {
  -40.0 * kDeg, -10.0 * kDeg, 20.0 * kDeg, 90.0 * kDeg, 88.0 * kDeg, 0.0, 0.0};

BuiltinAction waveLeftArm()
{
  BuiltinAction action;
  action.id = "action_04";
  action.label = "Wave left arm";
  action.description =
    "Home the left arm, then sweep joint3 between the start and end pose, until cancelled.";
  action.required_control_mode = kMotion;

  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    auto skill = ctx.skill;
    auto source = ctx.source;
    auto cancelled = ctx.cancelled;
    auto logger = ctx.node->get_logger();

    std::vector<double> home;
    try {
      home = source->loadWaypoint("homePoses/laHomeAngle");
    } catch (const std::exception& e) {
      done(false, std::string("could not read the left home waypoint: ") + e.what());
      return;
    }

    auto sweep = std::make_shared<std::function<void(bool)>>();
    auto cycle = std::make_shared<int>(0);

    *sweep = [skill, cancelled, logger, done, sweep, cycle](bool to_end) {
      if (cancelled()) {
        done(false, "cancelled");
        return;
      }
      RCLCPP_INFO(logger, "action_04: sweep %d (%s)", ++(*cycle), to_end ? "end" : "start");

      skill->moveToJoint(
        "left_arm", to_end ? kWaveLeftArmEnd : kWaveLeftArmStart, kWaveProfile, 0.0, 0.0,
        [sweep, to_end, done](bool ok, const std::string& error) {
          if (!ok) {
            done(false, error);
            return;
          }
          (*sweep)(!to_end);
        });
    };

    skill->moveToJoint(
      "left_arm", home, kWaveProfile, 0.0, 0.0,
      [cancelled, logger, done, sweep](bool ok, const std::string& error) {
        if (!ok) {
          done(false, "home move failed: " + error);
          return;
        }
        if (cancelled()) {
          done(false, "cancelled");
          return;
        }
        RCLCPP_INFO(logger, "action_04: left arm homed, starting sweep");
        (*sweep)(true);
      });
  };

  return action;
}

// ── action_05: wave right arm ────────────────────────────────────────────

// See kWaveLeftArmStart's note above (trimmed 8->7 for ee_type:=none).
const std::vector<double> kWaveRightArmStart = {
  40.0 * kDeg, 10.0 * kDeg, 20.0 * kDeg, 90.0 * kDeg, -88.0 * kDeg, 0.0, 0.0};
const std::vector<double> kWaveRightArmEnd = {
  40.0 * kDeg, 10.0 * kDeg, -20.0 * kDeg, 90.0 * kDeg, -88.0 * kDeg, 0.0, 0.0};

BuiltinAction waveRightArm()
{
  BuiltinAction action;
  action.id = "action_05";
  action.label = "Wave right arm";
  action.description =
    "Home the right arm, then sweep joint3 between the start and end pose, until cancelled.";
  action.required_control_mode = kMotion;

  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    auto skill = ctx.skill;
    auto source = ctx.source;
    auto cancelled = ctx.cancelled;
    auto logger = ctx.node->get_logger();

    std::vector<double> home;
    try {
      home = source->loadWaypoint("homePoses/raHomeAngle");
    } catch (const std::exception& e) {
      done(false, std::string("could not read the right home waypoint: ") + e.what());
      return;
    }

    auto sweep = std::make_shared<std::function<void(bool)>>();
    auto cycle = std::make_shared<int>(0);

    *sweep = [skill, cancelled, logger, done, sweep, cycle](bool to_end) {
      if (cancelled()) {
        done(false, "cancelled");
        return;
      }
      RCLCPP_INFO(logger, "action_05: sweep %d (%s)", ++(*cycle), to_end ? "end" : "start");

      skill->moveToJoint(
        "right_arm", to_end ? kWaveRightArmEnd : kWaveRightArmStart, kWaveProfile, 0.0, 0.0,
        [sweep, to_end, done](bool ok, const std::string& error) {
          if (!ok) {
            done(false, error);
            return;
          }
          (*sweep)(!to_end);
        });
    };

    skill->moveToJoint(
      "right_arm", home, kWaveProfile, 0.0, 0.0,
      [cancelled, logger, done, sweep](bool ok, const std::string& error) {
        if (!ok) {
          done(false, "home move failed: " + error);
          return;
        }
        if (cancelled()) {
          done(false, "cancelled");
          return;
        }
        RCLCPP_INFO(logger, "action_05: right arm homed, starting sweep");
        (*sweep)(true);
      });
  };

  return action;
}

// ── action_06: right arm start/end loop ──────────────────────────────────

// See kWaveLeftArmStart's note above (trimmed 8->7 for ee_type:=none).
const std::vector<double> kLoopRightArmStart = {
  17.0 * kDeg, 10.0 * kDeg, 5.0 * kDeg, 17.0 * kDeg, 5.0 * kDeg, 3.0 * kDeg, 17.0 * kDeg};
const std::vector<double> kLoopRightArmEnd = {
  0.0 * kDeg, 13.0 * kDeg, 21.0 * kDeg, 61.0 * kDeg, 74.0 * kDeg, 12.0 * kDeg, -7.0 * kDeg};

BuiltinAction loopRightArm()
{
  BuiltinAction action;
  action.id = "action_06";
  action.label = "Loop right arm";
  action.description =
    "Home the right arm, then loop between the start and end pose, until cancelled.";
  action.required_control_mode = kMotion;

  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    auto skill = ctx.skill;
    auto source = ctx.source;
    auto cancelled = ctx.cancelled;
    auto logger = ctx.node->get_logger();

    std::vector<double> home;
    try {
      home = source->loadWaypoint("homePoses/raHomeAngle");
    } catch (const std::exception& e) {
      done(false, std::string("could not read the right home waypoint: ") + e.what());
      return;
    }

    auto loop = std::make_shared<std::function<void(bool)>>();
    auto cycle = std::make_shared<int>(0);

    *loop = [skill, cancelled, logger, done, loop, cycle](bool to_end) {
      if (cancelled()) {
        done(false, "cancelled");
        return;
      }
      RCLCPP_INFO(logger, "action_06: loop %d (%s)", ++(*cycle), to_end ? "end" : "start");

      skill->moveToJoint(
        "right_arm", to_end ? kLoopRightArmEnd : kLoopRightArmStart, kWaveProfile, 0.0, 0.0,
        [loop, to_end, done](bool ok, const std::string& error) {
          if (!ok) {
            done(false, error);
            return;
          }
          (*loop)(!to_end);
        });
    };

    skill->moveToJoint(
      "right_arm", home, kWaveProfile, 0.0, 0.0,
      [cancelled, logger, done, loop](bool ok, const std::string& error) {
        if (!ok) {
          done(false, "home move failed: " + error);
          return;
        }
        if (cancelled()) {
          done(false, "cancelled");
          return;
        }
        RCLCPP_INFO(logger, "action_06: right arm homed, starting loop");
        (*loop)(true);
      });
  };

  return action;
}

// ── action_07: left arm start/end loop (mirror of action_06) ────────────
// Mirror rule matches action_04/action_05's laHomeAngle/raHomeAngle
// convention: joints 1,2,3,5,6,7 flip sign, joint4 keeps its sign.

// See kWaveLeftArmStart's note above (trimmed 8->7 for ee_type:=none).
const std::vector<double> kLoopLeftArmStart = {
  -17.0 * kDeg, -10.0 * kDeg, -5.0 * kDeg, 17.0 * kDeg, -5.0 * kDeg, -3.0 * kDeg, -17.0 * kDeg};
const std::vector<double> kLoopLeftArmEnd = {
  0.0 * kDeg, -13.0 * kDeg, -21.0 * kDeg, 61.0 * kDeg, -74.0 * kDeg, -12.0 * kDeg, 7.0 * kDeg};

BuiltinAction loopLeftArm()
{
  BuiltinAction action;
  action.id = "action_07";
  action.label = "Loop left arm";
  action.description =
    "Home the left arm, then loop between the start and end pose (mirror of action_06), "
    "until cancelled.";
  action.required_control_mode = kMotion;

  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    auto skill = ctx.skill;
    auto source = ctx.source;
    auto cancelled = ctx.cancelled;
    auto logger = ctx.node->get_logger();

    std::vector<double> home;
    try {
      home = source->loadWaypoint("homePoses/laHomeAngle");
    } catch (const std::exception& e) {
      done(false, std::string("could not read the left home waypoint: ") + e.what());
      return;
    }

    auto loop = std::make_shared<std::function<void(bool)>>();
    auto cycle = std::make_shared<int>(0);

    *loop = [skill, cancelled, logger, done, loop, cycle](bool to_end) {
      if (cancelled()) {
        done(false, "cancelled");
        return;
      }
      RCLCPP_INFO(logger, "action_07: loop %d (%s)", ++(*cycle), to_end ? "end" : "start");

      skill->moveToJoint(
        "left_arm", to_end ? kLoopLeftArmEnd : kLoopLeftArmStart, kWaveProfile, 0.0, 0.0,
        [loop, to_end, done](bool ok, const std::string& error) {
          if (!ok) {
            done(false, error);
            return;
          }
          (*loop)(!to_end);
        });
    };

    skill->moveToJoint(
      "left_arm", home, kWaveProfile, 0.0, 0.0,
      [cancelled, logger, done, loop](bool ok, const std::string& error) {
        if (!ok) {
          done(false, "home move failed: " + error);
          return;
        }
        if (cancelled()) {
          done(false, "cancelled");
          return;
        }
        RCLCPP_INFO(logger, "action_07: left arm homed, starting loop");
        (*loop)(true);
      });
  };

  return action;
}

}  // namespace

// ── action_09: show pose ─────────────────────────────────────────────────
// Static both-arms pose (no sweep/loop) - one move, then done. Order matches
// the "both_arms" SRDF group (<group name="left_arm"/><group name="right_arm"/>):
// 7 left joints followed by 7 right joints.
const std::vector<double> kShowPose = {
  0.0, -12.0 * kDeg, 0.0, 48.0 * kDeg, 0.0, 0.0, 0.0,
  0.0, 12.0 * kDeg, 0.0, 48.0 * kDeg, 0.0, 0.0, 0.0};

BuiltinAction showPose()
{
  BuiltinAction action;
  action.id = "action_09";
  action.label = "Show";
  action.description = "Move both arms to the show pose (joint2/joint4 raised).";
  action.required_control_mode = kMotion;

  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    auto skill = ctx.skill;

    // "safe_rrt" (OMPL), not "fast_ptp" (Pilz PTP) - see homeBothArms' note
    // above, Pilz PTP fails to plan for the composite "both_arms" group.
    skill->moveToJoint(
      "both_arms", kShowPose, "safe_rrt", 0.0, 0.0,
      [done](bool ok, const std::string& error) {
        if (!ok) {
          done(false, "show move failed: " + error);
          return;
        }
        done(true, "");
      });
  };

  return action;
}

// ── action_10/11/12: head rotate left / right / home (single-shot) ───────
// Same tilt-joint convention as headRotate() (action_03) - see that
// function's comment for why the "tilt" arg carries the left/right angle
// on this rig. Unlike action_03, these move once and finish (no sweep),
// matching /api/head's "left"/"right"/"home" shortcuts and duration.
constexpr double kHeadMoveDurationSec = 0.4;

BuiltinAction headRotateLeft()
{
  BuiltinAction action;
  action.id = "action_10";
  action.label = "Head rotate left";
  action.description = "Move the head to +10 degrees.";
  action.required_control_mode = sequence_executor::kModeAny;

  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    ctx.hand->setHead(0.0, kHeadRotateRad, kHeadMoveDurationSec,
      [done](bool ok, const std::string& error) { done(ok, error); });
  };

  return action;
}

BuiltinAction headRotateRight()
{
  BuiltinAction action;
  action.id = "action_11";
  action.label = "Head rotate right";
  action.description = "Move the head to -10 degrees.";
  action.required_control_mode = sequence_executor::kModeAny;

  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    ctx.hand->setHead(0.0, -kHeadRotateRad, kHeadMoveDurationSec,
      [done](bool ok, const std::string& error) { done(ok, error); });
  };

  return action;
}

BuiltinAction headRotateHome()
{
  BuiltinAction action;
  action.id = "action_12";
  action.label = "Head rotate home";
  action.description = "Move the head back to 0 degrees.";
  action.required_control_mode = sequence_executor::kModeAny;

  action.run = [](BuiltinContext& ctx, BuiltinAction::DoneCallback done) {
    ctx.hand->setHead(0.0, 0.0, kHeadMoveDurationSec,
      [done](bool ok, const std::string& error) { done(ok, error); });
  };

  return action;
}

}  // namespace

void registerQvicActions(BuiltinActionRegistry& registry)
{
  registry.add(homeBothArms());
  registry.add(waveLeftArmEllipse());
  registry.add(headRotate());
  registry.add(waveLeftArm());
  registry.add(waveRightArm());
  registry.add(loopRightArm());
  registry.add(loopLeftArm());
  registry.add(waveRightArmEllipse());
  registry.add(showPose());
  registry.add(headRotateLeft());
  registry.add(headRotateRight());
  registry.add(headRotateHome());
}

}  // namespace qvic_2026
