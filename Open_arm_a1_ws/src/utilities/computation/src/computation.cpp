// Empty placeholder for computation.cpp
#include "computation/computation.h"
#include "computation/definition.h"
#include "computation/conversion.h"
#include <iostream>


namespace utilities
{
namespace computation
{
    // --- JointBase ---
    std::string JointBase::toString() const
    {
        return "joints: [" + vecToString(joints) + "], name: " + joint_name + ", planning_group: " + joint_plangroup;
    }

    // --- Joint ---
    Joint::Joint()
    {
        joints = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        joint_name = "";
        joint_plangroup = "";
    }

    Joint::Joint(const std::vector<double>& joint_values, const bool isRadian, const std::string& planningGroup, const std::string& name)
    {
        joint_name = name;
        joint_plangroup = planningGroup;
        if (isRadian) {
            joints = joint_values;
        } else {
            joints = degreesToRadians(joint_values);
        }
    }

    void Joint::offset(const std::vector<double>& offset_values)
    {
        for (size_t i = 0; i < std::min(joints.size(), offset_values.size()); ++i) {
            joints[i] += offset_values[i];
        }
    }

    // --- HandJoints ---
    HandJoints::HandJoints()
    {
        joints = {0.0};
        joint_name = "";
        joint_plangroup = "";
    }

    HandJoints::HandJoints(std::vector<double> joint_values, const bool isRadian, const std::string& planningGroup, const std::string& name)
    {
        joint_name = name;
        joint_plangroup = planningGroup;
        if (isRadian) {
            joints = joint_values;
        } else {
            joints = degreesToRadians(joint_values);
        }
    }

    HandJoints::HandJoints(const std::string& planningGroup, const std::string& name, std::vector<double> angles, bool isRadian)
    {
        joint_name = name;
        joint_plangroup = planningGroup;
        if (isRadian) {
            joints = angles;
        } else {
            joints = degreesToRadians(angles);
        }
    }

    // --- HeadJoints ---
    HeadJoints::HeadJoints()
    {
        joints = {0.0, 0.0};
        joint_name = "";
        joint_plangroup = "";
    }

    HeadJoints::HeadJoints(const std::string& name, const std::vector<double>& angles, bool isRadian)
    {
        joint_name = name;
        joint_plangroup = "";
        if (isRadian) {
            joints = angles;
        } else {
            joints = degreesToRadians(angles);
        }
    }

    // --- PoseBase ---
    std::string PoseBase::toString() const
    {
        return "pos: " + tfvec3ToString(pos) + ", quat: " + tfQuatToString(quat) + ", name: " + pose_name + ", planning_group: " + pose_planninggroup;
    }

    std::vector<double> PoseBase::toVector() const
    {
        return poseQuatToVec(pos, quat);
    }

    geometry_msgs::msg::Pose PoseBase::toPoseMsg() const
    {
        return poseQuatToPoseMsgs(pos, quat);
    }

    geometry_msgs::msg::PoseStamped PoseBase::toPoseMsgsStamped() const
    {
        geometry_msgs::msg::PoseStamped stamped_pose;
        stamped_pose.header.frame_id = "base_link";
        stamped_pose.pose = toPoseMsg();
        return stamped_pose;
    }

    void PoseBase::reset()
    {
        pos = tf2::Vector3(0.0, 0.0, 0.0);
        quat = tf2::Quaternion(0.0, 0.0, 0.0, 1.0);
        pose_name = "";
        pose_planninggroup = "";
    }

    void PoseBase::setData(std::vector<double> pose_values, const std::string& planningGroup, const std::string& name)
    {
        pose_name = name;
        pose_planninggroup = planningGroup;
        if (pose_values.size() >= 7) {
            vecToPoseQuat(pose_values, pos, quat);
        } else if (pose_values.size() >= 3) {
            pos[0] = pose_values[0];
            pos[1] = pose_values[1];
            pos[2] = pose_values[2];
            if (pose_values.size() == 6) {
                quat.setRPY(pose_values[3], pose_values[4], pose_values[5]);
            } else {
                quat = tf2::Quaternion(0.0, 0.0, 0.0, 1.0);
            }
        }
    }

    // --- Pose ---
    Pose::Pose()
    {
        reset();
    }

    Pose::Pose(const tf2::Vector3& p, const tf2::Quaternion& q)
    {
        pos = p;
        quat = q;
        pose_name = "";
        pose_planninggroup = "";
    }

    Pose::Pose(double x, double y, double z, double qx, double qy, double qz, double qw)
    {
        pos = tf2::Vector3(x, y, z);
        quat = tf2::Quaternion(qx, qy, qz, qw);
        pose_name = "";
        pose_planninggroup = "";
    }

    tf2::Matrix3x3 Pose::getRotationMat() const
    {
        return tf2::Matrix3x3(quat);
    }

    tf2::Transform Pose::getTransform() const
    {
        return tf2::Transform(quat, pos);
    }

    void Pose::offsetPos(const tf2::Vector3& offset)
    {
        pos += offset;
    }

    void Pose::offsetQuat(const tf2::Quaternion& offset)
    {
        quat = offset * quat;
    }

    void Pose::offsetXYZ(const double x_offset, double y_offset, double z_offset)
    {
        pos[0] += x_offset;
        pos[1] += y_offset;
        pos[2] += z_offset;
    }

    void Pose::offsetRPY(const double roll_offset, const double pitch_offset, const double yaw_offset)
    {
        tf2::Quaternion offset;
        offset.setRPY(roll_offset, pitch_offset, yaw_offset);
        quat = offset * quat;
    }

    // --- PoseStamped ---
    PoseStamped::PoseStamped()
    {
        reset();
    }

    PoseStamped::PoseStamped(const std::vector<double>& pose_values, const std::string& planningGroup, const std::string& name)
    {
        setData(pose_values, planningGroup, name);
    }

    PoseStamped::PoseStamped(const tf2::Vector3& position, const tf2::Quaternion& orientation)
    {
        pos = position;
        quat = orientation;
        pose_name = "";
        pose_planninggroup = "";
    }

    // --- Helper conversion functions ---
    Pose tfTransformToPose(const tf2::Transform& tf)
    {
        return Pose(tf.getOrigin(), tf.getRotation());
    }

    tf2::Transform poseToTfTransform(const Pose& pose)
    {
        return tf2::Transform(pose.quat, pose.pos);
    }

    Pose poseMsgsToPose(const geometry_msgs::msg::Pose& pose_msg)
    {
        tf2::Vector3 pos(pose_msg.position.x, pose_msg.position.y, pose_msg.position.z);
        tf2::Quaternion quat(pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w);
        return Pose(pos, quat);
    }

    Pose multiplyTransformToPose(const tf2::Transform tf1, const tf2::Transform tf2)
    {
        tf2::Transform output_tf = tf1 * tf2;
        return(tfTransformToPose(output_tf));
    }

    Pose convertPoseInCamCoordinateToPose(const Pose& pose_in_cam ,const Pose& head_pose)
    {
        return multiplyTransformToPose(poseToTfTransform(head_pose), poseToTfTransform(pose_in_cam));
    }

    Pose convertPoseInBaseCoordinateToCam(const Pose& pose_in_base, const Pose& head_pose)
    {
        return multiplyTransformToPose(poseToTfTransform(head_pose), poseToTfTransform(pose_in_base));
    }

    Pose getAbsolutePose(const Pose& offset_pose, const Pose& base_pose)
    {
        return multiplyTransformToPose(poseToTfTransform(base_pose), poseToTfTransform(offset_pose));
    }

    Pose getAbsolutePose(const Pose& offset_pose, const geometry_msgs::msg::Pose& base_pose)
    {
        Pose base_pose_ = poseMsgsToPose(base_pose);
        return multiplyTransformToPose(poseToTfTransform(base_pose_), poseToTfTransform(offset_pose));
    }

    Pose getAbsolutePose(const double offset_x, const double offset_y, const double offset_z, const Pose& base_pose)
	{
		Pose output = base_pose;
		output.pos[0] += offset_x;
		output.pos[1] += offset_y;
		output.pos[2] += offset_z;
		return output;
	}
	Pose getAbsolutePose(const double offset_x, const double offset_y, const double offset_z, const geometry_msgs::msg::Pose& base_pose)
	{
		Pose output = poseMsgsToPose(base_pose);
		return getAbsolutePose(offset_x, offset_y, offset_z, output);
	}
	tf2::Vector3 getAbsolutePose(const tf2::Vector3& vector, const double distance, const Pose& base_pose)
	{
		tf2::Transform origin_tf(tf2::Quaternion(0, 0, 0, 1), tf2::Vector3(0, 0, 0));
		return origin_tf.inverse() * poseToTfTransform(base_pose) * (distance * vector.normalized());
	}
	Pose getOffsetPose(const Pose& poselink1tolink2, const Pose& poselink2tolink3)
	{
		return multiplyTransformToPose(poseToTfTransform(poselink1tolink2), poseToTfTransform(poselink2tolink3));
	}
	tf2::Vector3 movePoseAlongVector(const tf2::Vector3& vector, const double distance, const tf2::Vector3& base_pose)
	{
		return base_pose + (distance * vector.normalized());
	}
	tf2::Vector3 movePoseAlongVector(const tf2::Vector3& pose_begin, const tf2::Vector3& pose_end, const double distance,
									 const tf2::Vector3& base_pose)
	{
		tf2::Vector3 dir_vec = pose_end - pose_begin;
		return base_pose + (distance * dir_vec.normalized());
	}
	Pose movePoseAlongVector(const Pose& vector, const double distance, const Pose& base_pose)
	{
		Pose output = base_pose;
		output.pos		= movePoseAlongVector(vector.pos, distance, base_pose.pos);
		return output;
	}
	tf2::Transform getTransformBetweenPoses(const Pose& start_pose, const Pose& end_pose)
	{
		return poseToTfTransform(start_pose).inverse() * poseToTfTransform(end_pose);
	}
	// not in use
	Pose getControlPoseFromEff(const Pose& current_eef_pose, const Pose& current_tip_pose, const Pose& control_pose,
							   const Pose& placing_pose, bool orientation_base_on_tip)
	{
		Pose eef_pose;
		Pose control_pose_		 = getAbsolutePose(placing_pose, control_pose);
		tf2::Transform offset_matrix = getTransformBetweenPoses(current_tip_pose, current_eef_pose);
		Pose offset_pose_inv	 = tfTransformToPose(offset_matrix.inverse());
		Pose offset_pose		 = tfTransformToPose(offset_matrix);
		if(orientation_base_on_tip == false) {
			// orientation is relative of object and eef link
			eef_pose		   = getAbsolutePose(offset_pose_inv, control_pose_);
			control_pose_.quat = eef_pose.quat;
		}
		eef_pose = getAbsolutePose(offset_pose, control_pose_);
		return eef_pose;
	}
	Pose getOffsettedManipulatingObjectInv(const Pose& offset_matrix_pose, const Pose& control_pose,
											   const Pose& placing_pose, bool orientation_base_on_tip)
	{
		Pose eef_pose;
		Pose control_pose_ = getAbsolutePose(placing_pose, control_pose);
		tf2::Transform transform(offset_matrix_pose.quat, offset_matrix_pose.pos);
		Pose offset_pose_inv = tfTransformToPose(transform);
		Pose offset_pose	 = tfTransformToPose(transform.inverse());
		if(orientation_base_on_tip == false) {
			// orientation is relative of object and eef link
			eef_pose		  = getAbsolutePose(offset_pose_inv, control_pose_);
			control_pose_.quat = eef_pose.quat;
		}
		eef_pose = getAbsolutePose(offset_pose, control_pose_);
		return eef_pose;
	}
	
	std::vector<double> calculateCorrectedHeadJoints(const std::vector<double>& joints, const std::vector<std::vector<double>>& coef,
													 const std::vector<std::vector<double>>& power, const std::vector<double>& intercept)
	{
		if(coef.size() == 0 || power.size() == 0 || intercept.size() == 0) {
			std::cerr << " Invalid head_data.yaml data. Can not get corrected head joints!" << std::endl;
			return joints;
		}

		std::vector<double> result_joints = {0, 0};
		for (std::size_t i = 0; i < power.size(); ++i) {
			result_joints[0] += coef[0][i] * pow(joints[0], power[i][0]) * pow(joints[1], power[i][1]);
			result_joints[1] += coef[1][i] * pow(joints[0], power[i][0]) * pow(joints[1], power[i][1]);
		}

		result_joints[0] = result_joints[0] + intercept[0] + joints[0];
		result_joints[1] = result_joints[1] + intercept[1] + joints[1];
		printf("    Head joints [%f, %f] corrected as [%f, %f]\n", joints[0], joints[1], result_joints[0], result_joints[1]);
		fflush(stdout);
		return result_joints;
	}
	bool isPointInEllipse(const double centreX, const double centreY, const double halfWidth, const double halfHeight, const double x,
						  const double y)
	{
		return (pow(x - centreX, 2) / pow(halfWidth, 2) + pow(y - centreY, 2) / pow(halfHeight, 2)) <= 1;
	}
	bool isPointInEllipse(const std::vector<double> ellipse, const double x, const double y)
	{
		return isPointInEllipse(ellipse[0], ellipse[1], ellipse[2], ellipse[3], x, y);
	}
	PlaneParameter getPosePlaneParameter(const Pose& pose, const Plane plane)
	{
		tf2::Vector3 pos;
		Pose plane_pose;
		PlaneParameter plane_param;
		// Set plane pose
		if(plane == PLANE_YZ) {
			pos = tf2::Vector3(1, 0, 0);
		}
		else if(plane == PLANE_ZX) {
			pos = tf2::Vector3(0, 1, 0);
		}
		else if(plane == PLANE_XY) {
			pos = tf2::Vector3(0, 0, 1);
		}
		plane_pose.pos	= pos;
		plane_pose.quat = tf2::Quaternion(0, 0, 0, 1);
		// trasform the plane relative to pose
		Pose Transformed_plane = getAbsolutePose(plane_pose, pose);
		plane_param.origin		   = pose.pos;
		plane_param.normal_vector  = Transformed_plane.pos - pose.pos;
		return plane_param;
	}
} // namespace computation
} // namespace utilities