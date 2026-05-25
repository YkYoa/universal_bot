#include "computation/conversion.h"
#include "computation/computation.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace utilities 
{
namespace computation
{
	double degreesToRadians(double degrees) {
		return degrees * (M_PI / 180.0);
	}

	std::vector<double> degreesToRadians(const std::vector<double>& degrees) {
		std::vector<double> radians;
		radians.reserve(degrees.size());
		for (double d : degrees) {
			radians.push_back(degreesToRadians(d));
		}
		return radians;
	}

	double radiansToDegrees(double radians) {
		return radians * (180.0 / M_PI);
	}

	std::vector<double> radiansToDegrees(const std::vector<double>& radians) {
		std::vector<double> degrees;
		degrees.reserve(radians.size());
		for (double r : radians) {
			degrees.push_back(radiansToDegrees(r));
		}
		return degrees;
	}

	geometry_msgs::msg::Pose vecToPoseMsgs(const std::vector<double>& pose)
	{
		geometry_msgs::msg::Pose pose_msg;
		if(pose.size() < 7){
			std::cerr << "Error: Pose vector must contain 7 elements.\n";
			return pose_msg;
		}

		pose_msg.position.x = pose[0];
		pose_msg.position.y = pose[1];
		pose_msg.position.z = pose[2];
		pose_msg.orientation.x = pose[3];
		pose_msg.orientation.y = pose[4];
		pose_msg.orientation.z = pose[5];
		pose_msg.orientation.w = pose[6];
		return pose_msg;
	}

	std::string vecToString(const std::vector<double>& vec)
	{
		std::ostringstream output;
		for(auto v : vec)
			output << std::to_string(v) << ",";

		std::string outstr = output.str();
		if(!vec.empty() && outstr.size() >= 2)
			outstr.resize(outstr.size() - 2);

		return outstr;
	}

	std::string tfvec3ToString(const tf2::Vector3& vec)
	{
		std::ostringstream oss;
		oss << "(" << vec.x() << ", " << vec.y() << ", " << vec.z() << ")";
		return oss.str();
	}

	std::string tfQuatToString(const tf2::Quaternion& quat)
	{
		std::ostringstream oss;
		oss << "(" << quat.x() << ", " << quat.y() << ", " << quat.z() << ", " << quat.w() << ")";
		return oss.str();
	}

	std::vector<double> poseToVec(const tf2::Vector3& pos)
	{
		return {pos.x(), pos.y(), pos.z()};
	}
	std::vector<double> quatToVec(const tf2::Quaternion& quat)
	{
		return {quat.x(), quat.y(), quat.z(), quat.w()};
	}
	std::vector<double> poseQuatToVec(const tf2::Vector3& pos, const tf2::Quaternion& quat)
	{
		std::vector<double> output = poseToVec(pos);
		std::vector<double> q	   = quatToVec(quat);
		output.insert(output.end(), q.begin(), q.end());
		return output;
	}
	geometry_msgs::msg::Pose poseQuatToPoseMsgs(const tf2::Vector3& pos, const tf2::Quaternion& quat)
	{
		geometry_msgs::msg::Pose output;
		output.position.x	 = pos.x();
		output.position.y	 = pos.y();
		output.position.z	 = pos.z();
		output.orientation.x = quat.x();
		output.orientation.y = quat.y();
		output.orientation.z = quat.z();
		output.orientation.w = quat.w();
		return output;
	}
	void vecToPoseQuat(const std::vector<double>& input, tf2::Vector3& pos, tf2::Quaternion& quat)
	{
		if(input.size() < 7) {
			std::cerr << "vecTOPoseQuat: input vector size error" << std::endl;
			return;
		};
		pos[0]	= input[0];
		pos[1]	= input[1];
		pos[2]	= input[2];
		quat[0] = input[3];
		quat[1] = input[4];
		quat[2] = input[5];
		quat[3] = input[6];
	}
	Eigen::VectorXd vecToEigenVec(std::vector<double> vec)
	{
		return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
	}
	std::vector<double> eigenVecToVec(Eigen::VectorXd eigenVec)
	{
		return std::vector<double>(eigenVec.data(), eigenVec.data() + eigenVec.size());
	}
	
	Pose eigenIsoToPose(const Eigen::Isometry3d& eigen_iso)
	{
		Eigen::Vector3d position = eigen_iso.translation();
		Eigen::Quaterniond quat(eigen_iso.rotation());
		return Pose(position.x(), position.y(), position.z(), quat.x(), quat.y(), quat.z(), quat.w());
	}
	
	Pose poseStampedMsgsToPose(const geometry_msgs::msg::PoseStamped& pose_stamped)
	{
		return poseMsgsToPose(pose_stamped.pose);
	}
	geometry_msgs::msg::Pose poseToPoseMsgs(const Pose& pose)
	{
		return poseQuatToPoseMsgs(pose.pos, pose.quat);
	}
	tf2::Transform poseMsgsToTfTransform(const geometry_msgs::msg::Pose& pose_msg)
	{
		Pose pose = poseMsgsToPose(pose_msg);
		return poseToTfTransform(pose);
	}

	std::vector<double> quatToEuler(const tf2::Quaternion& quat)
	{
		std::vector<double> rpy;
		tf2::Matrix3x3 m(quat);
		double roll, pitch, yaw;
		m.getRPY(roll, pitch, yaw);
		rpy.push_back(roll);
		rpy.push_back(pitch);
		rpy.push_back(yaw);
		return rpy;
	}
	std::vector<double> quatToEuler(const double x, const double y, const double z, const double w)
	{
		tf2::Quaternion q(x, y, z, w);
		return quatToEuler(q);
	}
	tf2::Matrix3x3 eulerToTfMatrix(const std::vector<double>& euler_angles)
	{
		if(euler_angles.size() != 3) {
			std::cerr << "eulerToTfMatrix: input vector must contain exactly three Euler Angles, Given vector size" << euler_angles.size()
					  << std::endl;
			return tf2::Matrix3x3();
		}
		tf2::Matrix3x3 rotation_matrix;
		rotation_matrix.setRPY(euler_angles[0], euler_angles[1], euler_angles[2]);
		return rotation_matrix;
	}
	std::vector<double> tfMatrixToEuler(const tf2::Matrix3x3& rotation_matrix)
	{
		double r, p, y;
		rotation_matrix.getRPY(r, p, y);
		return {r, p, y};
	}
	tf2::Matrix3x3 quaternionToTfMatrix(const tf2::Quaternion& quat)
	{
		if(quat.length2() < 1e-6) {
			std::cerr << "quaternionToTfMatrix: Quaternion is too small and might not represent a valid rotation. Given q2"
					  << quat.length2() << std::endl;
			return tf2::Matrix3x3();
		}
		tf2::Quaternion q = quat;
		return tf2::Matrix3x3(q.normalized());
	}
	tf2::Quaternion tfMatrixToQuaternion(const tf2::Matrix3x3& rotation_matrix)
	{
		tf2::Quaternion q;
		rotation_matrix.getRotation(q);
		return q;
	}
	geometry_msgs::msg::Pose tfTransStampedToPoseMsgs(const geometry_msgs::msg::TransformStamped& transformStamped)
	{
		geometry_msgs::msg::Pose pose;
		pose.position.x	 = transformStamped.transform.translation.x;
		pose.position.y	 = transformStamped.transform.translation.y;
		pose.position.z	 = transformStamped.transform.translation.z;
		pose.orientation = transformStamped.transform.rotation;
		return pose;
	}
	tf2::Transform posRotToTfTransform(const tf2::Matrix3x3& rotation, const tf2::Vector3& translation)
	{
		tf2::Quaternion quaternion;
		rotation.getRotation(quaternion);					// Convert rotation matrix to quaternion
		tf2::Transform transform(quaternion, translation);	// Create tf2::Transform
		return transform;
	}
	tf2::Transform eigenToTfTransform(const Eigen::Matrix4d& eigen_mat)
	{
		// Extract rotation (3x3)
		tf2::Matrix3x3 tf2_rotation(eigen_mat(0, 0), eigen_mat(0, 1), eigen_mat(0, 2), eigen_mat(1, 0), eigen_mat(1, 1), eigen_mat(1, 2),
									eigen_mat(2, 0), eigen_mat(2, 1), eigen_mat(2, 2));
		// Extract translation (x, y, z)
		tf2::Vector3 tf2_translation(eigen_mat(0, 3), eigen_mat(1, 3), eigen_mat(2, 3));
		// Construct tf2::Transform
		return tf2::Transform(tf2_rotation, tf2_translation);
	}
	Eigen::Matrix4d transformStampedToMatrix(const geometry_msgs::msg::TransformStamped& transformStamped)
	{
		// Extract translation
		double tx = transformStamped.transform.translation.x;
		double ty = transformStamped.transform.translation.y;
		double tz = transformStamped.transform.translation.z;
		// Extract rotation (quaternion)
		tf2::Quaternion q(transformStamped.transform.rotation.x, transformStamped.transform.rotation.y,
						  transformStamped.transform.rotation.z, transformStamped.transform.rotation.w);
		// Convert quaternion to rotation matrix
		tf2::Matrix3x3 rotationMatrix(q);
		// Convert to Eigen 4x4 transformation matrix
		Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();	// Initialize as identity matrix
		for(int i = 0; i < 3; i++)
			for(int j = 0; j < 3; j++)
				mat(i, j) = rotationMatrix[i][j];  // Copy rotation
		mat(0, 3) = tx;
		mat(1, 3) = ty;
		mat(2, 3) = tz;
		return mat;
	}
} // namespace computation
} // namespace utilities
