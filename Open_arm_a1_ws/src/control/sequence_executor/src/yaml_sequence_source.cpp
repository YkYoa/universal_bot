#include "sequence_executor/yaml_sequence_source.hpp"

#include <stdexcept>

namespace sequence_executor {

namespace {

// Matches kHandMoveDurationS from the sequence interpreter this replaced.
constexpr double kHandMoveDurationS = 1.0;

const char* prefixForArm(const std::string& arm)
{
  return arm == "right_arm" ? "ra" : "la";
}

std::pair<std::string, std::string> splitRef(const std::string& ref)
{
  const auto slash = ref.find('/');
  if (slash == std::string::npos) {
    return {std::string(), ref};
  }
  return {ref.substr(0, slash), ref.substr(slash + 1)};
}

// One hand_pose step carrying whichever of the four vectors the section holds,
// so both hands move together - the same concurrent fan-out the old
// runHandPoseSection() did.
bool buildHandStep(SequenceYaml& yaml, const std::string& section, Step& step)
{
  const HandPose left = yaml.handPose(section, "lh");
  const HandPose right = yaml.handPose(section, "rh");
  if (!left.has_yaw && !left.has_flex && !right.has_yaw && !right.has_flex) {
    return false;
  }

  step.type = "hand_pose";
  step.name = "Hand (" + section + ")";
  step.required_control_mode = kModeAny;
  step.section = section;
  step.duration = kHandMoveDurationS;
  if (left.has_yaw) step.left_yaw = left.yaw;
  if (left.has_flex) step.left_flex = left.flex;
  if (right.has_yaw) step.right_yaw = right.yaw;
  if (right.has_flex) step.right_flex = right.flex;
  return true;
}

}  // namespace

YamlSequenceSource::YamlSequenceSource(const std::string& yaml_path)
  : yaml_path_(yaml_path), yaml_(std::make_shared<SequenceYaml>(yaml_path))
{
}

std::vector<std::string> YamlSequenceSource::listSequences()
{
  return yaml_->sequenceNames();
}

SequenceSpec YamlSequenceSource::loadSequence(const std::string& name)
{
  const SequenceDef def = yaml_->sequence(name);

  SequenceSpec spec;
  spec.name = name;
  spec.description = "from " + yaml_path_;
  spec.arm = def.arm;
  spec.planner_profile = def.planner_profile;
  spec.repeat = def.repeat;

  int index = 0;
  auto push = [&spec, &index](Step step) {
    step.index = index++;
    spec.steps.push_back(std::move(step));
  };

  if (!def.home_section.empty()) {
    const SectionSpeed speed = yaml_->speedForSection(def.home_section);
    Step home;
    home.type = "move_joint";
    home.name = "Home (" + def.home_section + ")";
    home.required_control_mode = kModeMotion;
    home.arm = def.arm;
    home.velocity = speed.velocity > 0.0 ? speed.velocity : 0.0;
    home.acceleration = speed.acceleration > 0.0 ? speed.acceleration : 0.0;

    if (def.arm == "both_arms") {
      const bool has_left = !yaml_->armAngle(def.home_section, "la").empty();
      const bool has_right = !yaml_->armAngle(def.home_section, "ra").empty();
      if (has_left && has_right) {
        home.waypoint = def.home_section + "/laHomeAngle";
        home.right_waypoint = def.home_section + "/raHomeAngle";
        push(home);
      }
    } else {
      const char* prefix = prefixForArm(def.arm);
      if (!yaml_->armAngle(def.home_section, prefix).empty()) {
        home.waypoint = def.home_section + "/" + prefix + "HomeAngle";
        push(home);
      }
    }

    Step hand;
    if (buildHandStep(*yaml_, def.home_section, hand)) {
      push(hand);
    }
  }

  if (!def.body_sections.empty()) {
    for (const auto& section : def.body_sections) {
      Step hand;
      if (buildHandStep(*yaml_, section, hand)) {
        push(hand);
      }
    }
  } else if (!def.body_section.empty()) {
    const SectionSpeed speed = yaml_->speedForSection(def.body_section);
    Step body;
    body.type = "move_joint_sequence";
    body.name = "Play " + def.body_section;
    body.required_control_mode = kModeMotion;
    body.arm = def.arm;
    body.section = def.body_section;
    body.right_section = def.body_right_section;
    body.exclude_points = def.exclude_points;
    body.velocity = speed.velocity > 0.0 ? speed.velocity : 0.0;
    body.acceleration = speed.acceleration > 0.0 ? speed.acceleration : 0.0;
    push(body);
  }

  if (spec.steps.empty()) {
    throw std::runtime_error("sequences:" + name + " produced no runnable steps");
  }

  // A sequence out of the YAML is always arm motion; there is no way to write
  // a hand-guiding step in that schema.
  for (const auto& step : spec.steps) {
    if (step.required_control_mode == kModeMotion) {
      spec.required_control_mode = kModeMotion;
      break;
    }
  }
  return spec;
}

std::vector<double> YamlSequenceSource::loadWaypoint(const std::string& ref)
{
  const auto [section, key] = splitRef(ref);
  if (section.empty()) {
    throw std::runtime_error("waypoint '" + ref + "' must be qualified as 'section/name'");
  }
  auto values = yaml_->value(section, key);
  if (values.empty()) {
    throw std::runtime_error("no waypoint '" + ref + "' in " + yaml_path_);
  }
  return values;
}

std::vector<std::vector<double>> YamlSequenceSource::loadSection(const std::string& section)
{
  return yaml_->waypoints(section);
}

bool YamlSequenceSource::hasWaypoint(const std::string& ref)
{
  const auto [section, key] = splitRef(ref);
  return !section.empty() && !yaml_->value(section, key).empty();
}

bool YamlSequenceSource::hasSection(const std::string& section)
{
  return yaml_->hasSection(section);
}

std::string YamlSequenceSource::describe() const
{
  return "sequence.yaml at " + yaml_path_;
}

}  // namespace sequence_executor
