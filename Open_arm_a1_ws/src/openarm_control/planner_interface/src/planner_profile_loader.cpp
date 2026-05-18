// ─────────────────────────────────────────────────────────────────────────────
// planner_profile_loader.cpp
// ─────────────────────────────────────────────────────────────────────────────

#include "planner_interface/planner_profile.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace planner_interface {

// ── Simple YAML value parser (no dependency on yaml-cpp) ─────────────────────
// This is a minimal parser for our flat profile YAML structure.
// For production, consider linking yaml-cpp if the config grows complex.

namespace {

/// Trim whitespace from both ends of a string.
std::string trim(const std::string & s) {
  auto start = s.find_first_not_of(" \t\r\n");
  auto end   = s.find_last_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  return s.substr(start, end - start + 1);
}

/// Remove surrounding quotes from a string value.
std::string unquote(const std::string & s) {
  auto t = trim(s);
  if (t.size() >= 2 &&
      ((t.front() == '"' && t.back() == '"') ||
       (t.front() == '\'' && t.back() == '\''))) {
    return t.substr(1, t.size() - 2);
  }
  return t;
}

}  // namespace

// ── PlannerProfileLoader ─────────────────────────────────────────────────────

PlannerProfileLoader::PlannerProfileLoader(const std::string & yaml_path)
: yaml_path_(yaml_path)
{
  load(yaml_path);
}

PlannerProfileLoader PlannerProfileLoader::from_package(const std::string & package_name)
{
  const auto share_dir = ament_index_cpp::get_package_share_directory(package_name);
  return PlannerProfileLoader(share_dir + "/config/planner_profiles.yaml");
}

void PlannerProfileLoader::load(const std::string & yaml_path)
{
  std::ifstream file(yaml_path);
  if (!file.is_open()) {
    throw std::runtime_error(
      "PlannerProfileLoader: cannot open " + yaml_path);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  profiles_.clear();
  default_profile_name_ = "safe_rrt";  // fallback default

  // State machine for parsing the simple YAML structure:
  //   profiles:
  //     profile_name:
  //       key: value
  //       ...
  //   default_profile: name

  std::string line;
  std::string current_profile;
  bool in_profiles_section = false;

  while (std::getline(file, line)) {
    // Skip comments and empty lines
    auto trimmed = trim(line);
    if (trimmed.empty() || trimmed[0] == '#') continue;

    // Count leading spaces to determine indentation level
    size_t indent = line.find_first_not_of(' ');
    if (indent == std::string::npos) continue;

    // Top-level keys (indent 0)
    if (indent == 0) {
      if (trimmed.find("profiles:") == 0) {
        in_profiles_section = true;
        current_profile.clear();
        continue;
      }
      if (trimmed.find("default_profile:") == 0) {
        in_profiles_section = false;
        auto colon = trimmed.find(':');
        default_profile_name_ = unquote(trimmed.substr(colon + 1));
        continue;
      }
      in_profiles_section = false;
      continue;
    }

    if (!in_profiles_section) continue;

    // Profile name (indent 2)
    if (indent == 2 && trimmed.back() == ':') {
      current_profile = trimmed.substr(0, trimmed.size() - 1);
      current_profile = trim(current_profile);
      profiles_[current_profile].name = current_profile;
      continue;
    }

    // Profile key-value (indent 4+)
    if (indent >= 4 && !current_profile.empty()) {
      auto colon = trimmed.find(':');
      if (colon == std::string::npos) continue;

      auto key   = trim(trimmed.substr(0, colon));
      auto value = unquote(trimmed.substr(colon + 1));
      auto & p   = profiles_[current_profile];

      if      (key == "description")          p.description = value;
      else if (key == "pipeline_id")          p.pipeline_id = value;
      else if (key == "planner_id")           p.planner_id  = value;
      else if (key == "velocity_scaling")     p.velocity_scaling     = std::stod(value);
      else if (key == "acceleration_scaling") p.acceleration_scaling = std::stod(value);
      else if (key == "planning_time")        p.planning_time        = std::stod(value);
      else if (key == "num_attempts")         p.num_attempts         = std::stoi(value);
    }
  }
}

std::optional<PlannerProfile> PlannerProfileLoader::get_profile(
  const std::string & name) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = profiles_.find(name);
  if (it == profiles_.end()) return std::nullopt;
  return it->second;
}

PlannerProfile PlannerProfileLoader::get_default_profile() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = profiles_.find(default_profile_name_);
  if (it != profiles_.end()) return it->second;

  // Ultimate fallback
  PlannerProfile fallback;
  fallback.name        = "safe_rrt";
  fallback.pipeline_id = "ompl";
  fallback.planner_id  = "RRTConnectkConfigDefault";
  return fallback;
}

std::vector<std::string> PlannerProfileLoader::list_profiles() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> names;
  names.reserve(profiles_.size());
  for (const auto & [name, _] : profiles_) {
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

void PlannerProfileLoader::reload()
{
  load(yaml_path_);
}

}  // namespace planner_interface
