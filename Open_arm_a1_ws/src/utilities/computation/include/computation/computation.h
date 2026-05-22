#pragma once 

#include <variant>
#include <vector>
#include <string>
#include "computation/definition.h"
#include "computation/conversion.h"

namespace utilities
{
namespace computation
{
    struct JointBase
    {
        std::vector<double> joints;
        std::string joint_name ="";
        std::string joint_plangroup ="";
        std::string toString() const;
    };

    struct Joint final : public JointBase
    {
        Joint();
        Joint(const std::vector<double>& joint_values, const bool isRadian = true, const std::string& planningGroup = "",
        const std::string& name ="");
        void offset(const std::vector<double>& offset);
    };

    struct HandJoints final : public JointBase
    {
        HandJoints();
        HandJoints(std::vector<double> joint_values, const bool isRadian = true, const std::string& planningGroup = "",
        const std::string& name = "");
        HandJoints(const std::string& planningGroup, const std::string& name, std::vector<double> angles, bool isRadian);
    };

    struct HeadJoints final : public JointBase
    {
        HeadJoints();
        HeadJoints(const std::string& name, const std::vector<double>& angles, bool isRadian = true);
    };

    struct PoseBase
    {
        tf2::Vector3 pos;
        tf2::Quaternion quat;
        std::string pose_name = "";
        std::string pose_planninggroup ="";
        std::string toString() const;
        std::vector<double> toVector() const;
        geometry_msgs::msg::Pose toPoseMsg() const;
        geometry_msgs::msg::PoseStamped toPoseMsgsStamped() const;
        void reset();
        void setData(std::vector<double> pose_valeus, const std::string& planningGroup ="", const std::string& name = "");
    };

    struct Pose final : public PoseBase
    {
        Pose();
        Pose(const tf2::Vector3& p, const tf2::Quaternion& q);
        Pose(double x, double y, double z, double qx = 0, double qy = 0, double qz = 0, double qw = 1);
        tf2::Matrix3x3 getRotationMat() const;
        tf2::Transform getTransform() const;
        void offsetPos(const tf2::Vector3& offset);
        void offsetQuat(const tf2::Quaternion& offset);
        void offsetXYZ(const double x_offset, double y_offset, double z_offset);
        void offsetRPY(const double roll_offset,const double pitch_offset, const double yaw_offset);
    };

    struct PoseStamped final : public PoseBase
    {
        PoseStamped();
        PoseStamped(const std::vector<double>& pose_values, const std::string& planningGroup ="", const std::string& name = "");
        PoseStamped(const tf2::Vector3& position, const tf2::Quaternion& orientation);
    };

    using KinematicObject = std::variant<Joint, Pose>;
    using KinematicObjects = std::vector<KinematicObject>;

    // Helper conversion functions
    Pose tfTransformToPose(const tf2::Transform& tf);
    tf2::Transform poseToTfTransform(const Pose& pose);
    Pose poseMsgsToPose(const geometry_msgs::msg::Pose& pose_msg);

    Pose multiplyTransformToPose(const tf2::Transform tf1, const tf2::Transform tf2);
    Pose convertPoseInCamCoordinateToPose(const Pose& pose_in_cam, const Pose& head_pose);
    Pose convertPoseInBaseCoordinateToCam(const Pose& pose_in_base, const Pose& head_pose);
    Pose getAbsolutePose(const Pose& offset_pose, const Pose& base_pose);
    Pose getAbsolutePose(const Pose& offset_pose, const geometry_msgs::msg::Pose& base_pose);
    Pose getAbsolutePose(const double offset_x, const double offset_y, const double offset_z, const Pose& base_pose);
    Pose getAbsolutePose(const double offset_x, const double offset_y, const double offset_z, const geometry_msgs::msg::Pose& base_pose);
    tf2::Vector3 getAbsolutePose(const tf2::Vector3& vector, const double distance, const Pose& base_pose);
    Pose getOffsetPose(const Pose& poselink1tolink2, const Pose& poselink2tolink3);
    tf2::Vector3 movePoseAlongVector(const tf2::Vector3& vector, const double distance, const tf2::Vector3& base_pose);
    tf2::Vector3 movePoseAlongVector(const tf2::Vector3& pose_begin, const tf2::Vector3& pose_end, const double distance,
									 const tf2::Vector3& base_pose);
    Pose movePoseAlongVector(const Pose& vector, const double distance, const Pose& base_pose);
    tf2::Transform getTransformBetweenPoses(const Pose& start_pose, const Pose& end_pose);
    Pose getControlPoseFromEff(const Pose& current_eef_pose, const Pose& current_tip_pose, const Pose& control_pose,
								   const Pose& placing_pose, bool orientation_base_on_tip);
	Pose getOffsettedManipulatingObjectInv(const Pose& offset_matrix_pose, const Pose& control_pose,
											   const Pose& placing_pose, bool orientation_base_on_tip);
	std::vector<double> calculateCorrectedHeadJoints(const std::vector<double>& joints, const std::vector<std::vector<double>>& coef,
													 const std::vector<std::vector<double>>& power, const std::vector<double>& intercept);
	bool isPointInEllipse(const std::vector<double> ellipse, const double x, const double y);
	bool isPointInEllipse(const double centreX, const double centreY, const double halfWidth, const double halfHeight, const double x,
						  const double y);
	PlaneParameter getPosePlaneParameter(const Pose& pose, const Plane plane);

} // namespace computation
} // namespace utilities