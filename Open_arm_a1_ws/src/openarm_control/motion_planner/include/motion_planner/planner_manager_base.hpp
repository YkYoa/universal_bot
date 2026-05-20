#ifndef __PLANNER_MANAGER_BASE_H__
#define __PLANNER_MANAGER_BASE_H__

#include <rclcpp/rclcpp.hpp>
#include "planning_interface/planner_request_response.hpp"

namespace motion_planner
{
    class PlannerManagerBase
    {
    public:
        PlannerManagerBase() = default;
        virtual ~PlannerManagerBase() = default;

        virtual bool initialize(const std::shared_ptr<rclcpp::Node>& node) = 0;
        virtual planning_interface::PlannerResponse plan(const planning_interface::PlannerRequest& request) = 0;
    };
}

#endif // __PLANNER_MANAGER_BASE_H__
