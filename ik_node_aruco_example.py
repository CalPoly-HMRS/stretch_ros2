#!/usr/bin/env python3

import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from rclpy.duration import Duration

from hello_misc import HelloNode, get_p1_to_p2_matrix
from ik_class import StretchIkRos

# Update this list to control which ArUco IDs are considered.
TARGET_ARUCO_IDS = [0, 2]

GRIPPER_OPEN = 0.6
GRIPPER_CLOSED = 0.2

class IkArucoExampleNode(HelloNode):
	def __init__(self):
		super().__init__()
		self.main("ik_aruco_example_node", "ik_aruco_example_node", wait_for_first_pointcloud=False)
		self.ik = StretchIkRos(self, tool_name="tool_stretch_dex_wrist")
		self.target_marker_pub = self.create_publisher(Marker, "ik_target_marker", 10)
		

	def _publish_line_marker(self, start_point, end_point, frame_id="base_link"):
		marker = Marker()
		marker.header.frame_id = frame_id
		marker.header.stamp = self.get_clock().now().to_msg()
		marker.ns = "ik_aruco_path"
		marker.id = 0
		marker.type = Marker.LINE_STRIP
		marker.action = Marker.ADD
		marker.scale.x = 0.01
		marker.color.r = 0.2
		marker.color.g = 0.4
		marker.color.b = 0.9
		marker.color.a = 0.9
		marker.lifetime = Duration(seconds=20.0).to_msg()

		start = Point()
		start.x = float(start_point[0])
		start.y = float(start_point[1])
		start.z = float(start_point[2])
		end = Point()
		end.x = float(end_point[0])
		end.y = float(end_point[1])
		end.z = float(end_point[2])
		marker.points = [start, end]

		self.target_marker_pub.publish(marker)

	def _publish_target_marker(self, target_point, frame_id="base_link"):
		marker = Marker()
		marker.header.frame_id = frame_id
		marker.header.stamp = self.get_clock().now().to_msg()
		marker.ns = "ik_aruco_target"
		marker.id = 0
		marker.type = Marker.SPHERE
		marker.action = Marker.ADD
		marker.pose.position.x = float(target_point[0])
		marker.pose.position.y = float(target_point[1])
		marker.pose.position.z = float(target_point[2])
		marker.pose.orientation.w = 1.0
		marker.scale.x = 0.05
		marker.scale.y = 0.05
		marker.scale.z = 0.05
		marker.color.r = 0.2
		marker.color.g = 0.8
		marker.color.b = 0.2
		marker.color.a = 0.9
		marker.lifetime = Duration(seconds=10.0).to_msg()
		self.target_marker_pub.publish(marker)

	def _find_first_aruco_target(self, base_frame="base_link"):
		if not TARGET_ARUCO_IDS:
			self.get_logger().warn("TARGET_ARUCO_IDS is empty; no tags to search for.")
			return None, None
		search_ids = list(TARGET_ARUCO_IDS)
		while rclpy.ok():
			time.sleep(2) # Wait 2 seconds before trying again, just cuz i felt like it
			for marker_id in search_ids:
				tag_frame = f"aruco_tag_{marker_id}"
				tag_to_base, _ = get_p1_to_p2_matrix(
					tag_frame,
					base_frame,
					self.tf2_buffer,
					timeout_s=0.05,
					verbose=False,
				)
				if tag_to_base is None:
					continue
				target_point = (tag_to_base @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
				return marker_id, target_point
			rclpy.spin_once(self, timeout_sec=0.1)
		return None, None

	def run_once(self):
		answer = input("Open gripper? [y/N]: ").strip().lower()
		if answer.startswith("y"):
			self.set_joint_poses([("stretch_gripper", GRIPPER_OPEN)])

		marker_id, target_point = self._find_first_aruco_target()
		if target_point is None:
			# self.get_logger().warn("No aruco_tag_* frame found in TF.")
			return

		q_init = self.ik.get_current_configuration(tool_name="tool_stretch_dex_wrist")
		current_point = self.ik.chain.forward_kinematics(q_init)[:3, 3]
		self._publish_target_marker(target_point)
		self._publish_line_marker(current_point, target_point)
		rclpy.spin_once(self, timeout_sec=0.1)

		answer = input(f"Move to aruco_tag_{marker_id}? [y/N]: ").strip().lower()
		if not answer.startswith("y"):
			self.get_logger().info("Skipping move.")
			return

		q_soln = self.ik.solve_point_ik(
			target_point,
			q_init=q_init,
			fixed_joints=["base_rotate", "base_translate"],
		)
		error = self.ik.compute_position_error(q_soln, target_point)
		self.get_logger().info(f"IK error: {error:.4f} m")

		if error < 0.5:
			self.ik.move_to_configuration(q_soln, tool_name="tool_stretch_dex_wrist")

			answer = input("Close gripper? [y/N]: ").strip().lower()
			if answer.startswith("y"):
				self.set_joint_poses([("stretch_gripper", GRIPPER_CLOSED)])
		else:
			self.get_logger().warn("IK solution outside tolerance")


def main():
	node = None
	try:
		node = IkArucoExampleNode()
		node.run_once()
	finally:
		if node is not None:
			node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
