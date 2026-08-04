#include "bt_executor/nodes/actions/execute_trajectory.hpp"
#include <iostream>

namespace bt_executor {

BT::NodeStatus ExecuteTrajectory::tick()
{
  // Skill execution is handled directly by PlanToPose / PlanToNamedPose, 
  // so this node immediately returns SUCCESS to maintain compatibility with the BT XML tree.
  std::cout << "[ExecuteTrajectory] Execution handled inline by Skill Server. Returning SUCCESS." << std::endl;
  return BT::NodeStatus::SUCCESS;
}

}  // namespace bt_executor
