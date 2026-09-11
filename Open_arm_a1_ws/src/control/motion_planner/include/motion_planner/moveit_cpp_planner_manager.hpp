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
    /// PlannerManagerBase backed by moveit_cpp::MoveItCpp: resolves a
    /// PlannerRequest (pose / joint target / joint sequence / Cartesian
    /// waypoints) against a loaded PlannerProfile, with on-disk plan caching,
    /// joint-sequence corner blending, and RViz marker/DisplayTrajectory
    /// publishing for visualization.
    class MoveItCppPlannerManager : public PlannerManagerBase
    {
    public:
        MoveItCppPlannerManager() = default;
        ~MoveItCppPlannerManager() override = default;

        /// Constructs MoveItCpp (with its planning scene monitor started) and
        /// loads planner profiles for the "motion_planner" package.
        bool initialize(const std::shared_ptr<rclcpp::Node>& node) override;
        /// Resolves `request` against a PlannerProfile: checks the on-disk plan
        /// cache first, then dispatches to pose/joint-target/joint-sequence/
        /// Cartesian-waypoint planning depending on which goal fields are set.
        planning_interface::PlannerResponse plan(const planning_interface::PlannerRequest& request) override;

        std::shared_ptr<moveit_cpp::MoveItCpp> getMoveItCpp() const { return moveit_cpp_; }

    private:
        /// End-effector link for `group`, resolved dynamically from the
        /// JointModelGroup's end-effector tips (varies with ee_type - see
        /// srdf_utils.py); returns "" if none is found.
        std::string ee_link_for_group(const std::string& group) const;
        /// Publishes (or deletes, if `visible` is false) an RViz sphere marker
        /// at `request`'s resolved target pose.
        void publish_target_marker_for_request(const planning_interface::PlannerRequest& request, bool visible);
        /// Publishes (or deletes) RViz sphere-list + line-strip markers tracing
        /// the end-effector path of `trajectory` for `group`.
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
