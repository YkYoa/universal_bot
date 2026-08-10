#pragma once
// -----------------------------------------------------------------------------
// scene_manager.hpp
//
// Planning-scene edits: collision objects, grasp attach/detach, and direct
// AllowedCollisionMatrix entries.
//
// This lives in robot_skills_node because that process already owns the
// PlanningSceneMonitor (through MoveItCppPlannerManager), and the scene has to
// be edited in the same process that plans against it - a separate node would
// race the monitor's own update stream.
//
// Deliberately a service, not another ExecuteSkill goal: an object id is not a
// motion parameter, and these calls complete in microseconds rather than
// needing feedback and cancellation.
//
// The grasp story: "attach" is the one to reach for. Attaching a collision
// object to a link makes MoveIt carry it with the arm, count it as part of the
// robot when avoiding everything else, and stop flagging it against the
// touch_links - which is what "allow collision while grasping" means in
// practice. Raw "allow" is for the other case: the tray or table the object was
// resting on.
// -----------------------------------------------------------------------------
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <openarm_messages/srv/scene_command.hpp>

#include "motion_planner/moveit_cpp_planner_manager.hpp"

namespace robot_skills {

class SceneManager
{
public:
  using SceneCommand = openarm_messages::srv::SceneCommand;

  SceneManager(rclcpp::Node::SharedPtr node,
               std::shared_ptr<motion_planner::MoveItCppPlannerManager> planner);

  // Advertises robot_skills_server/scene_command. Returns false if the planner
  // never came up with a PlanningSceneMonitor.
  bool start();

private:
  void handle(const std::shared_ptr<SceneCommand::Request> request,
              std::shared_ptr<SceneCommand::Response> response);

  // Each returns an empty string on success, or the reason it failed.
  std::string addObject(const SceneCommand::Request& req);
  std::string removeObject(const SceneCommand::Request& req);
  std::string attachObject(const SceneCommand::Request& req);
  std::string detachObject(const SceneCommand::Request& req);
  std::string setAllowed(const SceneCommand::Request& req, bool allowed);
  std::string clearScene();

  // Links allowed to touch a grasped object when the caller did not name any:
  // every link of the end effector `link` belongs to, so closing fingers on an
  // object is not reported as a collision. Falls back to just `link` itself.
  std::vector<std::string> defaultTouchLinks(const std::string& link) const;

  bool buildPrimitive(const SceneCommand::Request& req,
                      moveit_msgs::msg::CollisionObject& object,
                      std::string& error) const;

  // Pushes the edit out to /monitored_planning_scene so RViz, move_group, and
  // anything else watching see it.
  void publishSceneUpdate();

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<motion_planner::MoveItCppPlannerManager> planner_;
  rclcpp::Service<SceneCommand>::SharedPtr service_;
  rclcpp::Logger logger_;
};

}  // namespace robot_skills
