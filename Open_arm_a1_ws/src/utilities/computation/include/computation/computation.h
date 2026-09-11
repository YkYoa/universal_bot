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
    /// Common fields/behavior shared by every joint-space target type.
    struct JointBase
    {
        std::vector<double> joints;
        std::string joint_name ="";
        std::string joint_plangroup ="";
        /// "joints: [...], name: ..., planning_group: ...".
        std::string toString() const;
    };

    /// A 7-DOF arm joint target.
    struct Joint final : public JointBase
    {
        /// All-zero joints, no name/group.
        Joint();
        /// `joint_values` converted from degrees to radians first unless `isRadian`.
        Joint(const std::vector<double>& joint_values, const bool isRadian = true, const std::string& planningGroup = "",
        const std::string& name ="");
        /// Adds `offset[i]` to joints[i] for each shared index (extra entries
        /// on either side are ignored).
        void offset(const std::vector<double>& offset);
    };

    /// A gripper/hand joint target (single value by default).
    struct HandJoints final : public JointBase
    {
        /// Single joint at 0.0, no name/group.
        HandJoints();
        /// `joint_values` converted from degrees to radians first unless `isRadian`.
        HandJoints(std::vector<double> joint_values, const bool isRadian = true, const std::string& planningGroup = "",
        const std::string& name = "");
        /// Same as the other constructor with arguments reordered (name/group first).
        HandJoints(const std::string& planningGroup, const std::string& name, std::vector<double> angles, bool isRadian);
    };

    /// A 2-DOF head (pan/tilt) joint target.
    struct HeadJoints final : public JointBase
    {
        /// Both joints at 0.0, no name.
        HeadJoints();
        /// `angles` converted from degrees to radians first unless `isRadian`.
        HeadJoints(const std::string& name, const std::vector<double>& angles, bool isRadian = true);
    };

    /// Common fields/behavior shared by every Cartesian pose type.
    struct PoseBase
    {
        tf2::Vector3 pos;
        tf2::Quaternion quat;
        std::string pose_name = "";
        std::string pose_planninggroup ="";
        /// "pos: (...), quat: (...), name: ..., planning_group: ...".
        std::string toString() const;
        /// [x, y, z, qx, qy, qz, qw].
        std::vector<double> toVector() const;
        /// Converts to geometry_msgs/Pose.
        geometry_msgs::msg::Pose toPoseMsg() const;
        /// Converts to geometry_msgs/PoseStamped with frame_id "base_link".
        geometry_msgs::msg::PoseStamped toPoseMsgsStamped() const;
        /// Zeroes position, sets orientation to identity, and clears name/group.
        void reset();
        /// Sets pos/quat from a flat vector: 7 values = [x,y,z,qx,qy,qz,qw],
        /// 6 values = [x,y,z,roll,pitch,yaw], 3 values = position only
        /// (identity orientation). Also sets pose_name/pose_planninggroup.
        void setData(std::vector<double> pose_valeus, const std::string& planningGroup ="", const std::string& name = "");
    };

    /// A named/grouped Cartesian pose.
    struct Pose final : public PoseBase
    {
        /// Identity pose (reset()).
        Pose();
        Pose(const tf2::Vector3& p, const tf2::Quaternion& q);
        Pose(double x, double y, double z, double qx = 0, double qy = 0, double qz = 0, double qw = 1);
        /// Orientation as a 3x3 rotation matrix.
        tf2::Matrix3x3 getRotationMat() const;
        /// Position+orientation as a tf2::Transform.
        tf2::Transform getTransform() const;
        /// Adds `offset` to the position.
        void offsetPos(const tf2::Vector3& offset);
        /// Pre-multiplies the orientation by `offset` (offset applied in the
        /// parent/world frame, not the pose's own local frame).
        void offsetQuat(const tf2::Quaternion& offset);
        /// Adds each offset to the corresponding position component.
        void offsetXYZ(const double x_offset, double y_offset, double z_offset);
        /// Pre-multiplies the orientation by a quaternion built from the given RPY.
        void offsetRPY(const double roll_offset,const double pitch_offset, const double yaw_offset);
    };

    /// Same fields as Pose; a distinct type for call sites that specifically
    /// want a "stamped" pose value (see setData()'s 3/6/7-value convention).
    struct PoseStamped final : public PoseBase
    {
        /// Identity pose (reset()).
        PoseStamped();
        /// Forwards to PoseBase::setData().
        PoseStamped(const std::vector<double>& pose_values, const std::string& planningGroup ="", const std::string& name = "");
        PoseStamped(const tf2::Vector3& position, const tf2::Quaternion& orientation);
    };

    using KinematicObject = std::variant<Joint, Pose>;
    using KinematicObjects = std::vector<KinematicObject>;

    // Helper conversion functions
    /// tf2::Transform -> Pose.
    Pose tfTransformToPose(const tf2::Transform& tf);
    /// Pose -> tf2::Transform.
    tf2::Transform poseToTfTransform(const Pose& pose);
    /// geometry_msgs/Pose -> Pose.
    Pose poseMsgsToPose(const geometry_msgs::msg::Pose& pose_msg);

    /// Composes two transforms (tf1 * tf2) and returns the result as a Pose.
    Pose multiplyTransformToPose(const tf2::Transform tf1, const tf2::Transform tf2);
    /// Transforms a pose expressed in the camera frame into the frame
    /// `head_pose` is expressed in (head_pose * pose_in_cam).
    Pose convertPoseInCamCoordinateToPose(const Pose& pose_in_cam, const Pose& head_pose);
    /// Transforms a pose expressed in the base frame into the camera frame
    /// via `head_pose` (head_pose * pose_in_base) - same composition as
    /// convertPoseInCamCoordinateToPose(), named for the opposite direction
    /// of use.
    Pose convertPoseInBaseCoordinateToCam(const Pose& pose_in_base, const Pose& head_pose);
    /// Composes `offset_pose` onto `base_pose` (base_pose * offset_pose),
    /// i.e. offset_pose expressed relative to base_pose, transformed into
    /// base_pose's parent frame.
    Pose getAbsolutePose(const Pose& offset_pose, const Pose& base_pose);
    /// Same as the Pose/Pose overload, with `base_pose` given as a geometry_msgs/Pose.
    Pose getAbsolutePose(const Pose& offset_pose, const geometry_msgs::msg::Pose& base_pose);
    /// Adds a raw XYZ offset directly to `base_pose`'s position (orientation
    /// unchanged) - simpler than the transform-composing overloads above.
    Pose getAbsolutePose(const double offset_x, const double offset_y, const double offset_z, const Pose& base_pose);
    /// Same as the double/double/double/Pose overload, with `base_pose`
    /// given as a geometry_msgs/Pose.
    Pose getAbsolutePose(const double offset_x, const double offset_y, const double offset_z, const geometry_msgs::msg::Pose& base_pose);
    /// Moves a point `distance` along `vector` (normalized) from the world
    /// origin, expressed through `base_pose`'s transform.
    tf2::Vector3 getAbsolutePose(const tf2::Vector3& vector, const double distance, const Pose& base_pose);
    /// Composes two chained link-to-link poses (poselink1tolink2 * poselink2tolink3).
    Pose getOffsetPose(const Pose& poselink1tolink2, const Pose& poselink2tolink3);
    /// `base_pose` displaced by `distance` along the normalized direction `vector`.
    tf2::Vector3 movePoseAlongVector(const tf2::Vector3& vector, const double distance, const tf2::Vector3& base_pose);
    /// Same as the single-vector overload, with the direction given as
    /// `pose_end - pose_begin` instead of a pre-computed vector.
    tf2::Vector3 movePoseAlongVector(const tf2::Vector3& pose_begin, const tf2::Vector3& pose_end, const double distance,
									 const tf2::Vector3& base_pose);
    /// Pose overload: moves `base_pose`'s position by `distance` along
    /// `vector.pos`, keeping `base_pose`'s orientation.
    Pose movePoseAlongVector(const Pose& vector, const double distance, const Pose& base_pose);
    /// The transform from `start_pose` to `end_pose` (start_pose^-1 * end_pose).
    tf2::Transform getTransformBetweenPoses(const Pose& start_pose, const Pose& end_pose);
    /// Not currently used anywhere in this workspace. Computes a control
    /// pose for an end effector given its current tip/eef poses and a
    /// target control/placing pose, optionally re-basing orientation onto
    /// the tip frame - see the .cpp for the exact transform composition.
    Pose getControlPoseFromEff(const Pose& current_eef_pose, const Pose& current_tip_pose, const Pose& control_pose,
								   const Pose& placing_pose, bool orientation_base_on_tip);
	/// Same shape as getControlPoseFromEff() but takes the tip/eef offset
	/// directly as `offset_matrix_pose` instead of deriving it from two poses.
	Pose getOffsettedManipulatingObjectInv(const Pose& offset_matrix_pose, const Pose& control_pose,
											   const Pose& placing_pose, bool orientation_base_on_tip);
	/// Applies a polynomial correction (coef * joints[0]^power0 * joints[1]^power1,
	/// summed per term, plus intercept) to a 2-element [pan, tilt] head joint
	/// vector, per the calibration in head_data.yaml. Returns `joints`
	/// unchanged (with an error log) if any of coef/power/intercept is empty.
	std::vector<double> calculateCorrectedHeadJoints(const std::vector<double>& joints, const std::vector<std::vector<double>>& coef,
													 const std::vector<std::vector<double>>& power, const std::vector<double>& intercept);
	/// `ellipse` = [centreX, centreY, halfWidth, halfHeight]; forwards to the
	/// 5-argument overload.
	bool isPointInEllipse(const std::vector<double> ellipse, const double x, const double y);
	/// True if (x, y) lies on or inside the axis-aligned ellipse centered at
	/// (centreX, centreY) with the given half-width/half-height.
	bool isPointInEllipse(const double centreX, const double centreY, const double halfWidth, const double halfHeight, const double x,
						  const double y);
	/// The principal `plane` (XY/YZ/ZX) expressed in point-normal form after
	/// being carried along by `pose`'s transform.
	PlaneParameter getPosePlaneParameter(const Pose& pose, const Plane plane);

} // namespace computation
} // namespace utilities