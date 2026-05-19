#!/usr/bin/env python3

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from rclpy.duration import Duration

from hello_misc import HelloNode
from ik_class import StretchIkRos


class IkExampleNode(HelloNode):
    def __init__(self):
        super().__init__()
        self.main("ik_example_node", "ik_example_node", wait_for_first_pointcloud=False)
        self.ik = StretchIkRos(self)
        self.target_marker_pub = self.create_publisher(Marker, "ik_target_marker", 10)

    def _publish_line_marker(self, start_point, end_point, frame_id="base_link"):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ik_example_path"
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
        marker.ns = "ik_example"
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

    def run_once(self):
        # Example target pose (x, y, z) in base_link with a simple yaw.
        target_point = [-0.043, -0.441, 0.654]
        target_rpy = [0.0, 0.0, -np.pi / 2]
        target_pose = self.ik.make_target_pose(target_point, target_rpy)
        q_init = self.ik.get_current_configuration()
        current_point = self.ik.chain.forward_kinematics(q_init)[:3, 3]
        self._publish_target_marker(target_point)
        self._publish_line_marker(current_point, target_point)
        rclpy.spin_once(self, timeout_sec=0.1) # ensures it gets published

        # lets the user decide whether to move or not
        answer = input("Move to target? [y/N]: ").strip().lower()
        if not answer.startswith("y"):
            self.get_logger().info("Skipping move.")
            return

        q_soln = self.ik.solve_pose_ik(target_pose, q_init=q_init, fixed_joints=["base_rotate", "base_translate"])
        error = self.ik.compute_position_error(q_soln, target_point)
        self.get_logger().info(f"IK error: {error:.4f} m")

        if error < 0.5:
            self.ik.move_to_configuration(q_soln)
        else:
            self.get_logger().warn("IK solution outside tolerance")


def main():
    node = None
    try:
        node = IkExampleNode()
        node.run_once()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
