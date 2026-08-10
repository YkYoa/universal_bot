#include "sequence_executor/step_parser.hpp"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace sequence_executor {

namespace {

std::string context(const std::string& sequence_name, int index, const std::string& type)
{
  return sequence_name + " step " + std::to_string(index) + " (" + type + ")";
}

std::string str(const YAML::Node& params, const char* key)
{
  return params[key] ? params[key].as<std::string>() : std::string();
}

double num(const YAML::Node& params, const char* key, double fallback)
{
  return params[key] ? params[key].as<double>() : fallback;
}

bool flag(const YAML::Node& params, const char* key, bool fallback)
{
  return params[key] ? params[key].as<bool>() : fallback;
}

std::vector<double> doubles(const YAML::Node& params, const char* key)
{
  std::vector<double> out;
  if (!params[key] || !params[key].IsSequence()) {
    return out;
  }
  for (const auto& item : params[key]) {
    out.push_back(item.as<double>());
  }
  return out;
}

std::vector<int> ints(const YAML::Node& params, const char* key)
{
  std::vector<int> out;
  if (!params[key] || !params[key].IsSequence()) {
    return out;
  }
  for (const auto& item : params[key]) {
    out.push_back(item.as<int>());
  }
  return out;
}

std::vector<std::string> strings(const YAML::Node& params, const char* key)
{
  std::vector<std::string> out;
  if (!params[key] || !params[key].IsSequence()) {
    return out;
  }
  for (const auto& item : params[key]) {
    out.push_back(item.as<std::string>());
  }
  return out;
}

void requireLength(const std::vector<double>& values, std::size_t expected,
                   const std::string& where, const char* field)
{
  if (!values.empty() && values.size() != expected) {
    throw std::runtime_error(where + ": '" + field + "' needs " +
                             std::to_string(expected) + " values, got " +
                             std::to_string(values.size()));
  }
}

void requireNonEmpty(const std::string& value, const std::string& where, const char* field)
{
  if (value.empty()) {
    throw std::runtime_error(where + ": missing '" + field + "'");
  }
}

}  // namespace

Step parseStep(const std::string& sequence_name, int index, const std::string& name,
               const std::string& type, const std::string& params_json,
               const std::string& required_control_mode, bool enabled)
{
  const std::string where = context(sequence_name, index, type);

  YAML::Node params;
  try {
    params = params_json.empty() ? YAML::Node(YAML::NodeType::Map) : YAML::Load(params_json);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error(where + ": params are not valid JSON: " + e.what());
  }
  if (!params.IsMap()) {
    throw std::runtime_error(where + ": params must be a JSON object");
  }

  Step step;
  step.index = index;
  step.name = name.empty() ? type : name;
  step.type = type;
  step.required_control_mode =
    required_control_mode.empty() ? std::string(kModeAny) : required_control_mode;
  step.enabled = enabled;

  step.arm = str(params, "arm");
  step.waypoint = str(params, "waypoint");
  step.right_waypoint = str(params, "right_waypoint");
  step.positions = doubles(params, "positions");
  step.section = str(params, "section");
  step.right_section = str(params, "right_section");
  step.exclude_points = ints(params, "exclude_points");
  step.velocity = num(params, "velocity", 0.0);
  step.acceleration = num(params, "acceleration", 0.0);

  step.position = doubles(params, "position");
  step.orientation = doubles(params, "orientation");
  step.position_only = flag(params, "position_only", false);
  step.frame_id = str(params, "frame_id");

  step.group = str(params, "group");
  step.pose = str(params, "pose");

  step.left_yaw = doubles(params, "left_yaw");
  step.left_flex = doubles(params, "left_flex");
  step.right_yaw = doubles(params, "right_yaw");
  step.right_flex = doubles(params, "right_flex");
  step.duration = num(params, "duration", 1.0);

  step.side = str(params, "side");
  step.action = str(params, "action");
  step.seconds = num(params, "seconds", 0.0);

  step.object_id = str(params, "object_id");
  step.link = str(params, "link");
  step.primitive = str(params, "primitive");
  step.dimensions = doubles(params, "dimensions");
  step.touch_links = strings(params, "touch_links");
  step.links = strings(params, "links");

  // Only the checks that would otherwise turn into a crash or a wrong motion.
  // Full schema validation happens in store.py before the row is written.
  if (type == "move_joint") {
    if (step.waypoint.empty() && step.positions.empty()) {
      throw std::runtime_error(where + ": needs either 'waypoint' or 'positions'");
    }
    const std::size_t expected = (step.arm == "both_arms") ? 14u : 7u;
    requireLength(step.positions, expected, where, "positions");
  } else if (type == "move_joint_sequence") {
    requireNonEmpty(step.section, where, "section");
  } else if (type == "move_pose") {
    requireLength(step.position, 3, where, "position");
    requireLength(step.orientation, 4, where, "orientation");
    if (step.position.empty()) {
      throw std::runtime_error(where + ": missing 'position'");
    }
  } else if (type == "named_pose") {
    requireNonEmpty(step.group, where, "group");
    requireNonEmpty(step.pose, where, "pose");
  } else if (type == "hand_pose") {
    requireLength(step.left_yaw, 4, where, "left_yaw");
    requireLength(step.left_flex, 4, where, "left_flex");
    requireLength(step.right_yaw, 4, where, "right_yaw");
    requireLength(step.right_flex, 4, where, "right_flex");
    if (step.left_yaw.empty() && step.left_flex.empty() &&
        step.right_yaw.empty() && step.right_flex.empty()) {
      throw std::runtime_error(where + ": needs at least one of left/right yaw/flex");
    }
  } else if (type == "gripper") {
    requireNonEmpty(step.side, where, "side");
    requireNonEmpty(step.action, where, "action");
  } else if (type == "add_object") {
    requireNonEmpty(step.object_id, where, "object_id");
    requireNonEmpty(step.primitive, where, "primitive");
    requireLength(step.position, 3, where, "position");
  } else if (type == "remove_object" || type == "allow_collision" ||
             type == "disallow_collision" || type == "detach_object") {
    requireNonEmpty(step.object_id, where, "object_id");
  } else if (type == "attach_object") {
    requireNonEmpty(step.object_id, where, "object_id");
    requireNonEmpty(step.link, where, "link");
  }

  return step;
}

}  // namespace sequence_executor
