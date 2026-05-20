#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// planner_profile.hpp
//
// Data structures and loader for planner profiles.
//
// A PlannerProfile maps a human-readable name (e.g. "linear_approach") to
// concrete MoveIt parameters (pipeline_id, planner_id, velocity scaling, etc.).
//
// Usage:
//   auto loader = PlannerProfileLoader::from_package("motion_planner");
//   auto profile = loader.get_profile("fast_ptp");
//   // profile->pipeline_id == "pilz_industrial_motion_planner"
//   // profile->planner_id  == "PTP"
// ─────────────────────────────────────────────────────────────────────────────

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace planning_interface {

// ── PlannerProfile ────────────────────────────────────────────────────────────

struct PlannerProfile {
  std::string name;
  std::string description;
  std::string pipeline_id;         // "ompl" | "pilz_industrial_motion_planner"
                                   //  | "stomp" | "chomp"
  std::string planner_id;          // "RRTConnectkConfigDefault" | "PTP" | ...
  double      velocity_scaling     = 0.3;
  double      acceleration_scaling = 0.3;
  double      planning_time        = 5.0;
  int         num_attempts         = 10;
};

// ── PlannerProfileLoader ──────────────────────────────────────────────────────

class PlannerProfileLoader {
public:
  /// Construct from an explicit YAML file path.
  explicit PlannerProfileLoader(const std::string & yaml_path);

  /// Construct by looking up the config inside a ROS 2 package share directory.
  /// Equivalent to: PlannerProfileLoader(get_package_share("pkg") +
  ///                                     "/config/planner_profiles.yaml")
  static std::shared_ptr<PlannerProfileLoader> from_package(const std::string & package_name);

  /// Get a profile by name.  Returns std::nullopt if not found.
  std::optional<PlannerProfile> get_profile(const std::string & name) const;

  /// Get the default profile (from "default_profile" key in YAML).
  PlannerProfile get_default_profile() const;

  /// List all available profile names.
  std::vector<std::string> list_profiles() const;

  /// Reload profiles from disk (hot-reload support).
  void reload();

private:
  void load(const std::string & yaml_path);

  std::string yaml_path_;
  std::string default_profile_name_;
  mutable std::mutex mutex_;
  std::map<std::string, PlannerProfile> profiles_;
};

}  // namespace planning_interface
