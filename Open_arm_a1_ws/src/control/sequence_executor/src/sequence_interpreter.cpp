#include "sequence_executor/sequence_interpreter.hpp"

#include <algorithm>
#include <memory>

namespace sequence_executor {

namespace {
constexpr double kHandMoveDurationS = 1.0;
}  // namespace

SequenceInterpreter::SequenceInterpreter(
  rclcpp::Node::SharedPtr node, std::shared_ptr<SequenceYaml> yaml, std::shared_ptr<SkillClient> skill_client,
  std::shared_ptr<HandGripperClient> hand_client)
: node_(std::move(node)),
  yaml_(std::move(yaml)),
  skill_client_(std::move(skill_client)),
  hand_client_(std::move(hand_client)),
  logger_(rclcpp::get_logger("SequenceInterpreter"))
{
  status_pub_ = node_->create_publisher<std_msgs::msg::String>("~/status", 10);
  fault_pub_ = node_->create_publisher<std_msgs::msg::String>("~/fault", 10);
}

void SequenceInterpreter::publishStatus(const std::string& status)
{
  std_msgs::msg::String msg;
  msg.data = sequence_name_ + ": " + status;
  status_pub_->publish(msg);
  RCLCPP_INFO(logger_, "%s", msg.data.c_str());
}

void SequenceInterpreter::fail(const std::string& reason)
{
  std_msgs::msg::String msg;
  msg.data = sequence_name_ + ": " + reason;
  RCLCPP_ERROR(logger_, "HALTED: %s", msg.data.c_str());
  fault_pub_->publish(msg);
}

void SequenceInterpreter::run(const std::string& sequence_name)
{
  sequence_name_ = sequence_name;
  SequenceDef def;
  try {
    def = yaml_->sequence(sequence_name);
  } catch (const std::exception& e) {
    fail(std::string("failed to load sequence: ") + e.what());
    return;
  }

  publishStatus("starting");
  runHome(def, [this, def]() mutable {
    publishStatus("home complete, starting body");
    loopBody(def, def.repeat == 0 ? 1 : def.repeat);
  });
}

void SequenceInterpreter::runHome(const SequenceDef& def, std::function<void()> on_done)
{
  if (def.home_section.empty()) {
    on_done();
    return;
  }

  auto afterArm = [this, def, on_done]() {
    runHandPoseSection(def.home_section, kHandMoveDurationS, [this, on_done](bool ok) {
      if (!ok) {
        return;  // fail() already called inside runHandPoseSection
      }
      on_done();
    });
  };

  std::vector<double> joint_targets;
  std::string arm_for_call = def.arm;

  if (def.arm == "both_arms") {
    auto la = yaml_->armAngle(def.home_section, "la");
    auto ra = yaml_->armAngle(def.home_section, "ra");
    if (la.empty() || ra.empty()) {
      RCLCPP_WARN(
        logger_, "home_section '%s' missing laHomeAngle/raHomeAngle for both_arms - skipping arm home step",
        def.home_section.c_str());
      afterArm();
      return;
    }
    joint_targets = la;
    joint_targets.insert(joint_targets.end(), ra.begin(), ra.end());
  } else {
    const std::string side_prefix = (def.arm == "right_arm") ? "ra" : "la";
    joint_targets = yaml_->armAngle(def.home_section, side_prefix);
    if (joint_targets.empty()) {
      RCLCPP_WARN(
        logger_, "home_section '%s' missing %sHomeAngle for %s - skipping arm home step", def.home_section.c_str(),
        side_prefix.c_str(), def.arm.c_str());
      afterArm();
      return;
    }
  }

  skill_client_->moveToJoint(
    arm_for_call, joint_targets, def.planner_profile, 0.0, 0.0, [this, afterArm](bool ok, const std::string& err) {
      if (!ok) {
        fail("home arm step failed: " + err);
        return;
      }
      afterArm();
    });
}

void SequenceInterpreter::runHandPoseSection(
  const std::string& section, double duration, std::function<void(bool ok)> on_done)
{
  HandPose left = yaml_->handPose(section, "lh");
  HandPose right = yaml_->handPose(section, "rh");

  // Yaw and flex fire independently, not just as an all-or-nothing pair -
  // a body_sections step (e.g. handOpen) deliberately declares only Flex
  // keys, meaning "touch flex, leave yaw wherever it already is." Requiring
  // both together (matching home_section's own stricter "both must be set"
  // validation) would silently skip flex-only sections entirely.
  const int pending_count = (left.has_yaw ? 1 : 0) + (left.has_flex ? 1 : 0) + (right.has_yaw ? 1 : 0) +
                             (right.has_flex ? 1 : 0);
  if (pending_count == 0) {
    on_done(true);
    return;
  }

  // Fire whichever of {left,right}x{yaw,flex} are present concurrently;
  // on_done fires once after all of them complete. A shared failed-flag
  // makes sure fail() is reported exactly once and on_done(false) only
  // fires once even if multiple concurrent calls fail.
  auto pending = std::make_shared<int>(pending_count);
  auto failed = std::make_shared<bool>(false);

  auto onOne = [this, pending, failed, on_done](bool ok, const std::string& err) {
    if (!ok && !*failed) {
      *failed = true;
      fail("hand pose step failed: " + err);
    }
    if (--(*pending) == 0) {
      on_done(!*failed);
    }
  };

  if (left.has_yaw) {
    hand_client_->setHandYaw(
      "left_arm", left.yaw, duration, [onOne](bool ok, const std::string& err) { onOne(ok, err); });
  }
  if (left.has_flex) {
    hand_client_->setHandFlex(
      "left_arm", left.flex, duration, [onOne](bool ok, const std::string& err) { onOne(ok, err); });
  }
  if (right.has_yaw) {
    hand_client_->setHandYaw(
      "right_arm", right.yaw, duration, [onOne](bool ok, const std::string& err) { onOne(ok, err); });
  }
  if (right.has_flex) {
    hand_client_->setHandFlex(
      "right_arm", right.flex, duration, [onOne](bool ok, const std::string& err) { onOne(ok, err); });
  }
}

void SequenceInterpreter::loopBody(SequenceDef def, int remaining)
{
  if (remaining == 0) {
    publishStatus("done");
    return;
  }
  runBodyOnce(def, [this, def, remaining](bool ok) mutable {
    if (!ok) {
      return;  // fail() already called
    }
    const int next = (remaining == -1) ? -1 : remaining - 1;
    loopBody(def, next);
  });
}

void SequenceInterpreter::runBodyOnce(const SequenceDef& def, std::function<void(bool ok)> on_done)
{
  if (!def.body_sections.empty()) {
    runBodySectionsStep(def, 0, on_done);
    return;
  }

  std::vector<std::vector<double>> left_wp;
  try {
    left_wp = yaml_->waypoints(def.body_section);
  } catch (const std::exception& e) {
    fail(std::string("body_section: ") + e.what());
    on_done(false);
    return;
  }

  auto excluded = [&def](std::size_t index_0based) {
    const int point_number = static_cast<int>(index_0based) + 1;  // 1-indexed, matches exclude_points' convention
    return std::find(def.exclude_points.begin(), def.exclude_points.end(), point_number) != def.exclude_points.end();
  };

  std::vector<double> flat;
  if (!def.body_right_section.empty()) {
    std::vector<std::vector<double>> right_wp;
    try {
      right_wp = yaml_->waypoints(def.body_right_section);
    } catch (const std::exception& e) {
      fail(std::string("body_right_section: ") + e.what());
      on_done(false);
      return;
    }
    if (left_wp.size() != right_wp.size()) {
      fail(
        "body_section/body_right_section waypoint count mismatch: " + std::to_string(left_wp.size()) + " vs " +
        std::to_string(right_wp.size()));
      on_done(false);
      return;
    }
    for (std::size_t i = 0; i < left_wp.size(); ++i) {
      if (excluded(i)) {
        continue;
      }
      flat.insert(flat.end(), left_wp[i].begin(), left_wp[i].end());
      flat.insert(flat.end(), right_wp[i].begin(), right_wp[i].end());
    }
  } else {
    for (std::size_t i = 0; i < left_wp.size(); ++i) {
      if (excluded(i)) {
        continue;
      }
      flat.insert(flat.end(), left_wp[i].begin(), left_wp[i].end());
    }
  }

  const SectionSpeed speed = yaml_->speedForSection(def.body_section);
  skill_client_->moveToJointSequence(
    def.arm, flat, def.planner_profile, speed.velocity >= 0.0 ? speed.velocity : 0.0,
    speed.acceleration >= 0.0 ? speed.acceleration : 0.0, [this, on_done](bool ok, const std::string& err) {
      if (!ok) {
        fail("body joint_sequence failed: " + err);
        on_done(false);
        return;
      }
      on_done(true);
    });
}

void SequenceInterpreter::runBodySectionsStep(SequenceDef def, std::size_t index, std::function<void(bool ok)> on_done)
{
  if (index >= def.body_sections.size()) {
    on_done(true);
    return;
  }
  const std::string section = def.body_sections[index];
  runHandPoseSection(section, kHandMoveDurationS, [this, def, index, on_done](bool ok) mutable {
    if (!ok) {
      on_done(false);
      return;
    }
    runBodySectionsStep(def, index + 1, on_done);
  });
}

}  // namespace sequence_executor
