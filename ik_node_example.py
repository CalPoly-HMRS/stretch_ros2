#!/usr/bin/env python3

import numpy as np
import rclpy

from hello_misc import HelloNode
from ik_class import StretchIkRos


class IkExampleNode(HelloNode):
    def __init__(self):
        super().__init__()
        self.main("ik_example_node", "ik_example_node", wait_for_first_pointcloud=False)
        self.ik = StretchIkRos(self)

    def run_once(self):
        # Example target pose (x, y, z) in base_link with a simple yaw.
        target_point = [-0.043, -0.441, 0.654]
        target_rpy = [0.0, 0.0, -np.pi / 2]
        target_pose = self.ik.make_target_pose(target_point, target_rpy)

        q_init = self.ik.get_current_configuration()
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
