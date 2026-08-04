#ifndef __MOVEIT_CPP_PLANNER_MANAGER_H__
#define __MOVEIT_CPP_PLANNER_MANAGER_H__

#include "motion_planner/planner_manager_base.hpp"
#include <moveit/moveit_cpp/moveit_cpp.hpp>
#include <moveit/moveit_cpp/planning_component.hpp>
#include "planning_interface/planner_profile.hpp"
#include <visualization_msgs/msg/marker.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <memory>
#include <string>

namespace motion_planner
{
    class MoveItCppPlannerManager : public PlannerManagerBase
    {
    public:
        MoveItCppPlannerManager() = default;
        ~MoveItCppPlannerManager() override = default;

        bool initialize(const std::shared_ptr<rclcpp::Node>& node) override;
        planning_interface::PlannerResponse plan(const planning_interface::PlannerRequest& request) override;

        std::shared_ptr<moveit_cpp::MoveItCpp> getMoveItCpp() const { return moveit_cpp_; }

    private:
        std::string ee_link_for_group(const std::string& group) const;
        void publish_target_marker_for_request(const planning_interface::PlannerRequest& request, bool visible);
        void publish_trajectory_markers(const moveit_msgs::msg::RobotTrajectory& trajectory, const std::string& group, bool visible);

        std::shared_ptr<rclcpp::Node> node_;
        std::shared_ptr<moveit_cpp::MoveItCpp> moveit_cpp_;
        std::shared_ptr<planning_interface::PlannerProfileLoader> profile_loader_;
        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr target_marker_pub_;
        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr trajectory_marker_pub_;
        rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_pub_;
    };
}

#endif // __MOVEIT_CPP_PLANNER_MANAGER_H__
