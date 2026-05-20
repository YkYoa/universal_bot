#include "planning_interface/planner_profile.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "yaml-cpp/yaml.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace planning_interface {

PlannerProfileLoader::PlannerProfileLoader(const std::string & yaml_path)
  : yaml_path_(yaml_path)
{
  load(yaml_path_);
}

std::shared_ptr<PlannerProfileLoader> PlannerProfileLoader::from_package(const std::string & package_name)
{
  std::string share_dir = ament_index_cpp::get_package_share_directory(package_name);
  std::string yaml_path = share_dir + "/config/planner_profiles.yaml";
  return std::make_shared<PlannerProfileLoader>(yaml_path);
}

void PlannerProfileLoader::load(const std::string & yaml_path)
{
  std::lock_guard<std::mutex> lock(mutex_);
  profiles_.clear();

  YAML::Node config;
  try {
    config = YAML::LoadFile(yaml_path);
  } catch (const std::exception & e) {
    std::cerr << "[PlannerProfileLoader] Failed to load yaml file: " << yaml_path 
              << ", error: " << e.what() << std::endl;
    return;
  }

  if (config["default_profile"]) {
    default_profile_name_ = config["default_profile"].as<std::string>();
  }

  if (config["profiles"] && config["profiles"].IsMap()) {
    for (auto it = config["profiles"].begin(); it != config["profiles"].end(); ++it) {
      std::string name = it->first.as<std::string>();
      YAML::Node node = it->second;

      PlannerProfile profile;
      profile.name = name;
      if (node["description"]) {
        profile.description = node["description"].as<std::string>();
      }
      if (node["pipeline_id"]) {
        profile.pipeline_id = node["pipeline_id"].as<std::string>();
      }
      if (node["planner_id"]) {
        profile.planner_id = node["planner_id"].as<std::string>();
      }
      if (node["velocity_scaling"]) {
        profile.velocity_scaling = node["velocity_scaling"].as<double>();
      }
      if (node["acceleration_scaling"]) {
        profile.acceleration_scaling = node["acceleration_scaling"].as<double>();
      }
      if (node["planning_time"]) {
        profile.planning_time = node["planning_time"].as<double>();
      }
      if (node["num_attempts"]) {
        profile.num_attempts = node["num_attempts"].as<int>();
      }

      profiles_[name] = profile;
    }
  }
}

std::optional<PlannerProfile> PlannerProfileLoader::get_profile(const std::string & name) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = profiles_.find(name);
  if (it != profiles_.end()) {
    return it->second;
  }
  return std::nullopt;
}

PlannerProfile PlannerProfileLoader::get_default_profile() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!default_profile_name_.empty()) {
    auto it = profiles_.find(default_profile_name_);
    if (it != profiles_.end()) {
      return it->second;
    }
  }

  // Fallback default profile if not defined or not found
  PlannerProfile fallback;
  fallback.name = "default_fallback";
  fallback.pipeline_id = "ompl";
  fallback.planner_id = "RRTConnectkConfigDefault";
  fallback.velocity_scaling = 0.3;
  fallback.acceleration_scaling = 0.3;
  fallback.planning_time = 5.0;
  fallback.num_attempts = 10;
  return fallback;
}

std::vector<std::string> PlannerProfileLoader::list_profiles() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> list;
  for (const auto & pair : profiles_) {
    list.push_back(pair.first);
  }
  return list;
}

void PlannerProfileLoader::reload()
{
  load(yaml_path_);
}

}  // namespace planning_interface
