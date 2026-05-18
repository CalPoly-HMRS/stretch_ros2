import os

import ikpy.chain
import ikpy.utils.geometry
import numpy as np
import stretch_body.hello_utils as hu
import urdfpy


LINKS_TO_REMOVE = [
	"link_right_wheel",
	"link_left_wheel",
	"caster_link",
	"link_gripper_finger_left",
	"link_gripper_fingertip_left",
	"link_gripper_finger_right",
	"link_gripper_fingertip_right",
	"link_head",
	"link_head_pan",
	"link_head_tilt",
	"link_aruco_right_base",
	"link_aruco_left_base",
	"link_aruco_shoulder",
	"link_aruco_top_wrist",
	"link_aruco_inner_wrist",
	"camera_bottom_screw_frame",
	"camera_link",
	"camera_depth_frame",
	"camera_depth_optical_frame",
	"camera_infra1_frame",
	"camera_infra1_optical_frame",
	"camera_infra2_frame",
	"camera_infra2_optical_frame",
	"camera_color_frame",
	"camera_color_optical_frame",
	"camera_accel_frame",
	"camera_accel_optical_frame",
	"camera_gyro_frame",
	"camera_gyro_optical_frame",
	"laser",
	"respeaker_base",
]

JOINTS_TO_REMOVE = [
	"joint_right_wheel",
	"joint_left_wheel",
	"caster_joint",
	"joint_gripper_finger_left",
	"joint_gripper_fingertip_left",
	"joint_gripper_finger_right",
	"joint_gripper_fingertip_right",
	"joint_head",
	"joint_head_pan",
	"joint_head_tilt",
	"joint_aruco_right_base",
	"joint_aruco_left_base",
	"joint_aruco_shoulder",
	"joint_aruco_top_wrist",
	"joint_aruco_inner_wrist",
	"camera_joint",
	"camera_link_joint",
	"camera_depth_joint",
	"camera_depth_optical_joint",
	"camera_infra1_joint",
	"camera_infra1_optical_joint",
	"camera_infra2_joint",
	"camera_infra2_optical_joint",
	"camera_color_joint",
	"camera_color_optical_joint",
	"camera_accel_joint",
	"camera_accel_optical_joint",
	"camera_gyro_joint",
	"camera_gyro_optical_joint",
	"joint_laser",
	"joint_respeaker",
]


class StretchIkController:
	def __init__(self, robot, cartesian_urdf_path=None, urdf_path=None):
		self.robot = robot
		self.cartesian_urdf_path = cartesian_urdf_path or "/tmp/iktutorial/stretch.urdf"
		self.urdf_path = urdf_path or self._get_urdf_path()
		self.chain = self._make_chain(self.urdf_path, self.cartesian_urdf_path)

	@staticmethod
	def _get_urdf_path():
		return str(
			(
				hu.get_fleet_directory()
				/ "exported_urdf"
				/ "stretch.urdf"
			).absolute()
		)

	@staticmethod
	def _make_chain(urdf_path, cartesian_urdf_path):
		StretchIkController._build_cartesian_urdf(urdf_path, cartesian_urdf_path)
		return ikpy.chain.Chain.from_urdf_file(cartesian_urdf_path)

	@staticmethod
	def _build_cartesian_urdf(urdf_path, out_path):
		original_urdf = urdfpy.URDF.load(urdf_path)
		modified_urdf = original_urdf.copy()

		links_to_remove = [
			link for link in modified_urdf._links if link.name in LINKS_TO_REMOVE
		]
		for link in links_to_remove:
			modified_urdf._links.remove(link)

		joints_to_remove = [
			joint for joint in modified_urdf._joints if joint.name in JOINTS_TO_REMOVE
		]
		for joint in joints_to_remove:
			modified_urdf._joints.remove(joint)

		joint_base_translation = urdfpy.Joint(
			name="joint_base_translation",
			parent="base_link",
			child="link_base_translation",
			joint_type="prismatic",
			axis=np.array([1.0, 0.0, 0.0]),
			origin=np.eye(4, dtype=np.float64),
			limit=urdfpy.JointLimit(
				effort=100.0,
				velocity=1.0,
				lower=-1.0,
				upper=1.0,
			),
		)
		modified_urdf._joints.append(joint_base_translation)

		link_base_translation = urdfpy.Link(
			name="link_base_translation",
			inertial=None,
			visuals=None,
			collisions=None,
		)
		modified_urdf._links.append(link_base_translation)

		for joint in modified_urdf._joints:
			if joint.name == "joint_mast":
				joint.parent = "link_base_translation"

		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		modified_urdf.save(out_path)

	@staticmethod
	def _bound_range(chain, name, value):
		names = [link.name for link in chain.links]
		index = names.index(name)
		bounds = chain.links[index].bounds
		return min(max(value, bounds[0]), bounds[1])

	def get_current_configuration(self, tool=None):
		tool = tool or self.robot.end_of_arm.name
		if tool == "tool_stretch_gripper":
			q_base = 0.0
			q_lift = self._bound_range(
				self.chain, "joint_lift", self.robot.lift.status["pos"]
			)
			q_arml = self._bound_range(
				self.chain, "joint_arm_l0", self.robot.arm.status["pos"] / 4.0
			)
			q_yaw = self._bound_range(
				self.chain,
				"joint_wrist_yaw",
				self.robot.end_of_arm.status["wrist_yaw"]["pos"],
			)
			return [
				0.0,
				q_base,
				0.0,
				q_lift,
				0.0,
				q_arml,
				q_arml,
				q_arml,
				q_arml,
				q_yaw,
				0.0,
				0.0,
			]

		if tool == "tool_stretch_dex_wrist":
			q_base = 0.0
			q_lift = self._bound_range(
				self.chain, "joint_lift", self.robot.lift.status["pos"]
			)
			q_arml = self._bound_range(
				self.chain, "joint_arm_l0", self.robot.arm.status["pos"] / 4.0
			)
			q_yaw = self._bound_range(
				self.chain,
				"joint_wrist_yaw",
				self.robot.end_of_arm.status["wrist_yaw"]["pos"],
			)
			q_pitch = self._bound_range(
				self.chain,
				"joint_wrist_pitch",
				self.robot.end_of_arm.status["wrist_pitch"]["pos"],
			)
			q_roll = self._bound_range(
				self.chain,
				"joint_wrist_roll",
				self.robot.end_of_arm.status["wrist_roll"]["pos"],
			)
			return [
				0.0,
				q_base,
				0.0,
				q_lift,
				0.0,
				q_arml,
				q_arml,
				q_arml,
				q_arml,
				q_yaw,
				0.0,
				q_pitch,
				q_roll,
				0.0,
				0.0,
			]

		raise ValueError(f"Unsupported tool: {tool}")

	def move_to_configuration(self, q, tool=None):
		tool = tool or self.robot.end_of_arm.name
		if tool == "tool_stretch_gripper":
			q_base = q[1]
			q_lift = q[3]
			q_arm = q[5] + q[6] + q[7] + q[8]
			q_yaw = q[9]
			self.robot.base.translate_by(q_base)
			self.robot.lift.move_to(q_lift)
			self.robot.arm.move_to(q_arm)
			self.robot.end_of_arm.move_to("wrist_yaw", q_yaw)
			self.robot.push_command()
			return

		if tool == "tool_stretch_dex_wrist":
			q_base = q[1]
			q_lift = q[3]
			q_arm = q[5] + q[6] + q[7] + q[8]
			q_yaw = q[9]
			q_pitch = q[11]
			q_roll = q[12]
			self.robot.base.translate_by(q_base)
			self.robot.lift.move_to(q_lift)
			self.robot.arm.move_to(q_arm)
			self.robot.end_of_arm.move_to("wrist_yaw", q_yaw)
			self.robot.end_of_arm.move_to("wrist_pitch", q_pitch)
			self.robot.end_of_arm.move_to("wrist_roll", q_roll)
			self.robot.push_command()
			return

		raise ValueError(f"Unsupported tool: {tool}")

	def solve_point_ik(self, target_point, q_init=None):
		q_init = q_init or self.get_current_configuration()
		return self.chain.inverse_kinematics(target_point, initial_position=q_init)

	def make_target_pose(self, target_point, rpy):
		pose = np.eye(4)
		pose[:3, :3] = ikpy.utils.geometry.rpy_matrix(*rpy)
		pose[:3, 3] = np.array(target_point)
		return pose

	def solve_pose_ik(self, target_pose, q_init=None, pretarget_pose=None):
		q_init = q_init or self.get_current_configuration()
		if pretarget_pose is not None:
			q_init = self.chain.inverse_kinematics_frame(
				pretarget_pose,
				orientation_mode="all",
				initial_position=q_init,
			)
		return self.chain.inverse_kinematics_frame(
			target_pose,
			orientation_mode="all",
			initial_position=q_init,
		)

	def compute_position_error(self, q_soln, target_point):
		return np.linalg.norm(
			self.chain.forward_kinematics(q_soln)[:3, 3] - target_point
		)

