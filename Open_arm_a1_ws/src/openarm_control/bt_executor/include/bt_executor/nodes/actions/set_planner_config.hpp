#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// set_planner_config.hpp
//
// Synchronous action node: loads a named planner profile and writes the
// resolved MoveIt parameters to the blackboard.  Subsequent MoveToPose
// nodes read these keys to select the correct planning pipeline.
//
// This is the BT-side interface to the planner_interface library.
//
// Input ports:
//   profile   std::string  Profile name from planner_profiles.yaml
//                          e.g. "fast_ptp", "linear_approach", "safe_rrt"
//
// Blackboard writes:
//   BB_PIPELINE_ID       std::string
//   BB_PLANNER_ID        std::string
//   BB_VEL_SCALE         double
//   BB_ACC_SCALE         double
//   BB_PLAN_TIME         double
//   BB_NUM_ATTEMPTS      int
//   BB_PLANNER_PROFILE   std::string
//
// XML usage:
//   <SetPlannerConfig profile="linear_approach" />
//   <MoveToPose arm="{plan_active_arm}" target_pose="{vla_grasp_pose}" ... />
// ─────────────────────────────────────────────────────────────────────────────
#include "behaviortree_cpp/behavior_tree.h"
#include "rclcpp/rclcpp.hpp"
#include "planning_interface/planner_profile.hpp"
#include "bt_executor/blackboard_keys.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
namespace bt_executor {

class SetPlannerConfig : public BT::SyncActionNode
{
public:
  SetPlannerConfig(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config)
  {
    // Load profiles once at construction time.
    // The loader is shared across all SetPlannerConfig instances.
    static auto loader = planning_interface::PlannerProfileLoader::from_package("motion_planner");
    loader_ = loader;
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::string>("profile",
        "Named planner profile from planner_profiles.yaml"),
    };
  }

  BT::NodeStatus tick() override
  {
    std::string profile_name;
    if (!getInput("profile", profile_name) || profile_name.empty()) {
      // No profile specified — use default
      auto p = loader_->get_default_profile();
      apply_profile(p);
      return BT::NodeStatus::SUCCESS;
    }

    auto opt = loader_->get_profile(profile_name);
    if (!opt.has_value()) {
      RCLCPP_WARN(rclcpp::get_logger("SetPlannerConfig"),
        "Unknown profile '%s' — falling back to default",
        profile_name.c_str());
      auto p = loader_->get_default_profile();
      apply_profile(p);
      return BT::NodeStatus::SUCCESS;
    }

    apply_profile(opt.value());
    return BT::NodeStatus::SUCCESS;
  }

private:
  void apply_profile(const planning_interface::PlannerProfile & p)
  {
    auto & bb = *config().blackboard;
    bb.set(BB_PIPELINE_ID,     p.pipeline_id);
    bb.set(BB_PLANNER_ID,      p.planner_id);
    bb.set(BB_VEL_SCALE,       p.velocity_scaling);
    bb.set(BB_ACC_SCALE,       p.acceleration_scaling);
    bb.set(BB_PLAN_TIME,       p.planning_time);
    bb.set(BB_NUM_ATTEMPTS,    p.num_attempts);
    bb.set(BB_PLANNER_PROFILE, p.name);

    RCLCPP_INFO(rclcpp::get_logger("SetPlannerConfig"),
      "Planner → %s  (%s / %s  vel=%.1f)",
      p.name.c_str(), p.pipeline_id.c_str(), p.planner_id.c_str(),
      p.velocity_scaling);
  }

  std::shared_ptr<planning_interface::PlannerProfileLoader> loader_;
};

}  // namespace bt_executor
