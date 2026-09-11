#ifndef __PLANNER_MANAGER_BASE_H__
#define __PLANNER_MANAGER_BASE_H__

#include <rclcpp/rclcpp.hpp>
#include "planning_interface/planner_request_response.hpp"

namespace motion_planner
{
    /// Abstract planner backend: something that can turn a PlannerRequest into
    /// a PlannerResponse. MoveItCppPlannerManager is the only implementation.
    class PlannerManagerBase
    {
    public:
        PlannerManagerBase() = default;
        virtual ~PlannerManagerBase() = default;

        /// One-time setup against the owning node (e.g. building a planning
        /// scene monitor); returns false on unrecoverable init failure.
        virtual bool initialize(const std::shared_ptr<rclcpp::Node>& node) = 0;
        /// Plans a trajectory satisfying `request` (pose, joint target, joint
        /// sequence, or Cartesian waypoints - whichever it specifies).
        virtual planning_interface::PlannerResponse plan(const planning_interface::PlannerRequest& request) = 0;
    };
}

#endif // __PLANNER_MANAGER_BASE_H__
