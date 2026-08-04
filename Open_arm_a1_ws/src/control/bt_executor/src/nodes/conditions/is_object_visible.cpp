// is_object_visible.cpp
#include "bt_executor/nodes/conditions/is_object_visible.hpp"

namespace bt_executor {

BT::NodeStatus IsObjectVisible::tick()
{
  // Read the object label this node was instantiated with
  std::string label;
  getInput("object_label", label);

  // TODO: Replace this stub with a real world-model lookup.
  //
  // The VLA perception bridge should maintain a map of
  //   { object_label → geometry_msgs::PoseStamped }
  // on the blackboard (or a separate ROS topic).  Here you would:
  //   1. Read that map from the blackboard.
  //   2. Return SUCCESS if `label` is in the map and its timestamp is fresh.
  //   3. Return FAILURE otherwise.
  //
  // Stub: always return SUCCESS — the VLA model handles perception
  // via camera images, so we trust that the object is visible when
  // the system is running.
  return BT::NodeStatus::SUCCESS;
}

}  // namespace bt_executor
