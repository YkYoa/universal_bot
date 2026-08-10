#include "sequence_executor/sequence_yaml.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>

#include <yaml-cpp/yaml.h>

namespace sequence_executor {

namespace {

std::string toStringValue(const YAML::Node& node)
{
  if (node.IsScalar()) {
    return node.as<std::string>();
  }
  if (node.IsSequence()) {
    std::string out;
    for (std::size_t i = 0; i < node.size(); ++i) {
      if (i > 0) out += ", ";
      out += node[i].as<std::string>();
    }
    return out;
  }
  return "";
}

std::vector<std::string> parseTokens(const std::string& raw)
{
  std::vector<std::string> tokens;
  std::stringstream ss(raw);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(0, item.find_first_not_of(" \t\n\r"));
    item.erase(item.find_last_not_of(" \t\n\r") + 1);
    if (!item.empty()) {
      tokens.push_back(item);
    }
  }
  return tokens;
}

std::vector<double> parseDoubles(const std::string& raw)
{
  std::vector<double> out;
  for (const auto& tok : parseTokens(raw)) {
    out.push_back(std::stod(tok));
  }
  return out;
}

bool isJointAngleKey(const std::string& key)
{
  std::string k = key;
  std::transform(k.begin(), k.end(), k.begin(), ::tolower);
  return k.find("angle") != std::string::npos;
}

YAML::Node loadRoot(const std::string& yaml_path)
{
  std::ifstream f(yaml_path);
  if (!f.good()) {
    throw std::runtime_error("sequence.yaml not found: " + yaml_path);
  }
  f.close();
  YAML::Node root = YAML::LoadFile(yaml_path);
  if (root.IsNull() || !root.IsMap()) {
    throw std::runtime_error("sequence.yaml is empty or not a map: " + yaml_path);
  }
  return root;
}

YAML::Node requireSection(const YAML::Node& root, const std::string& section)
{
  if (!root[section] || !root[section].IsMap()) {
    throw std::runtime_error("section '" + section + "' not found in sequence.yaml");
  }
  return root[section];
}

}  // namespace

SequenceYaml::SequenceYaml(const std::string& yaml_path) : yaml_path_(yaml_path)
{
  // Just validate it loads at construction time; every accessor below
  // re-reads it (this only ever runs at startup, not a control-rate hot
  // path, so re-parsing per call trades a little redundant I/O for not
  // needing to store a YAML::Node - and therefore not needing yaml-cpp - in
  // this class's public header).
  loadRoot(yaml_path_);
}

SequenceDef SequenceYaml::sequence(const std::string& name) const
{
  YAML::Node root = loadRoot(yaml_path_);
  if (!root["sequences"] || !root["sequences"].IsMap() || !root["sequences"][name]) {
    throw std::runtime_error("sequences:" + name + " not found in sequence.yaml");
  }
  YAML::Node node = root["sequences"][name];

  SequenceDef def;
  if (node["arm"]) def.arm = node["arm"].as<std::string>();
  if (node["planner_profile"]) def.planner_profile = node["planner_profile"].as<std::string>();
  if (node["home_section"]) def.home_section = node["home_section"].as<std::string>();
  if (node["body_section"]) def.body_section = node["body_section"].as<std::string>();
  if (node["body_right_section"]) def.body_right_section = node["body_right_section"].as<std::string>();
  if (node["body_sections"]) {
    def.body_sections = parseTokens(toStringValue(node["body_sections"]));
  }
  if (node["repeat"]) def.repeat = node["repeat"].as<int>();
  if (node["exclude_points"]) {
    for (const auto& tok : parseTokens(toStringValue(node["exclude_points"]))) {
      def.exclude_points.push_back(std::stoi(tok));
    }
  }

  if (def.arm.empty()) {
    throw std::runtime_error("sequences:" + name + " missing required field 'arm'");
  }
  if (def.body_section.empty() && def.body_sections.empty()) {
    throw std::runtime_error("sequences:" + name + " needs either 'body_section' or 'body_sections'");
  }
  return def;
}

std::vector<std::string> SequenceYaml::sequenceNames() const
{
  YAML::Node root = loadRoot(yaml_path_);
  std::vector<std::string> names;
  if (!root["sequences"] || !root["sequences"].IsMap()) {
    return names;
  }
  for (auto it = root["sequences"].begin(); it != root["sequences"].end(); ++it) {
    names.push_back(it->first.as<std::string>());
  }
  return names;
}

std::vector<double> SequenceYaml::value(const std::string& section, const std::string& key) const
{
  YAML::Node root = loadRoot(yaml_path_);
  if (!root[section] || !root[section].IsMap() || !root[section][key]) {
    return {};
  }
  return parseDoubles(toStringValue(root[section][key]));
}

bool SequenceYaml::hasSection(const std::string& section) const
{
  YAML::Node root = loadRoot(yaml_path_);
  return root[section] && root[section].IsMap();
}

std::vector<double> SequenceYaml::armAngle(const std::string& section, const std::string& side_prefix) const
{
  YAML::Node root = loadRoot(yaml_path_);
  YAML::Node sec = requireSection(root, section);
  const std::string key = side_prefix + "HomeAngle";
  if (!sec[key]) {
    return {};
  }
  return parseDoubles(toStringValue(sec[key]));
}

HandPose SequenceYaml::handPose(const std::string& section, const std::string& side_prefix) const
{
  YAML::Node root = loadRoot(yaml_path_);
  YAML::Node sec = requireSection(root, section);

  const std::regex yaw_re("^" + side_prefix + ".*Yaw$");
  const std::regex flex_re("^" + side_prefix + ".*Flex$");

  HandPose pose;
  for (auto it = sec.begin(); it != sec.end(); ++it) {
    const std::string key = it->first.as<std::string>();
    if (std::regex_match(key, yaw_re)) {
      pose.has_yaw = true;
      pose.yaw = parseDoubles(toStringValue(it->second));
    } else if (std::regex_match(key, flex_re)) {
      pose.has_flex = true;
      pose.flex = parseDoubles(toStringValue(it->second));
    }
  }
  return pose;
}

std::vector<std::vector<double>> SequenceYaml::waypoints(const std::string& section) const
{
  YAML::Node root = loadRoot(yaml_path_);
  YAML::Node sec = requireSection(root, section);

  std::vector<std::vector<double>> out;
  for (auto it = sec.begin(); it != sec.end(); ++it) {
    const std::string key = it->first.as<std::string>();
    if (!isJointAngleKey(key)) {
      continue;
    }
    out.push_back(parseDoubles(toStringValue(it->second)));
  }
  if (out.empty()) {
    throw std::runtime_error("section '" + section + "' has no *Angle waypoint values");
  }
  return out;
}

SectionSpeed SequenceYaml::speedForSection(const std::string& section) const
{
  YAML::Node root = loadRoot(yaml_path_);
  SectionSpeed speed;
  if (!root["speed"] || !root["speed"].IsMap() || !root["speed"][section]) {
    return speed;
  }
  auto tokens = parseTokens(toStringValue(root["speed"][section]));
  if (tokens.empty()) {
    return speed;
  }
  try {
    speed.velocity = std::stod(tokens[0]);
  } catch (...) {
  }
  if (tokens.size() > 1) {
    try {
      speed.acceleration = std::stod(tokens[1]);
    } catch (...) {
    }
  }
  return speed;
}

}  // namespace sequence_executor
