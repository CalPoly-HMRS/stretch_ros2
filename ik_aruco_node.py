#!/usr/bin/env python3

import threading

import numpy as np

from std_msgs.msg import Int32MultiArray

from hello_misc import HelloNode
from ik_class import StretchIkRos


class IkExampleNode(HelloNode):
    def __init__(self):
        super().__init__()
        self.main("ik_example_node", "ik_example_node", wait_for_first_pointcloud=False)
        self.ik = StretchIkRos(self)
        self.target_tag_ids = [1]
        self.tag_frame_prefix = "aruco_tag_"
        self._prompt_in_progress = False
        self.create_subscription(
            Int32MultiArray,
            "aruco_tags_detected",
            self._aruco_tags_callback,
            10,
        )

    def _aruco_tags_callback(self, msg):
        tag_ids = [int(tag_id) for tag_id in msg.data]
        target_id = next(
            (tag_id for tag_id in tag_ids if tag_id in self.target_tag_ids),
            None,
        )
        if target_id is None:
            return
        if self._prompt_in_progress:
            return

        self._prompt_in_progress = True
        threading.Thread(
            target=self._prompt_move_to_tag,
            args=(target_id,),
            daemon=True,
        ).start()

    def _prompt_move_to_tag(self, tag_id):
        tf_msg = self._lookup_tag_tf(tag_id)
        if tf_msg is None:
            self._prompt_in_progress = False
            return

        translation = tf_msg.transform.translation
        target_point = [translation.x, translation.y, translation.z]
        self.get_logger().info(
            f"Tag {tag_id} at x={target_point[0]:.3f}, y={target_point[1]:.3f}, z={target_point[2]:.3f}"
        )

        answer = input(f"Move toward tag {tag_id}? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            self._move_to_target_point(target_point)
        else:
            self.get_logger().info("Skipping move to tag.")

        self._prompt_in_progress = False

    def _lookup_tag_tf(self, tag_id):
        tag_frame = f"{self.tag_frame_prefix}{tag_id}"
        tf_msg = self.get_tf("base_link", tag_frame)
        if tf_msg is None:
            self.get_logger().warn(f"No TF for tag frame {tag_frame}")
        return tf_msg

    def _move_to_target_point(self, target_point):
        q_init = self.ik.get_current_configuration()
        q_soln = self.ik.solve_point_ik(target_point, q_init=q_init)
        error = self.ik.compute_position_error(q_soln, target_point)
        self.get_logger().info(f"IK error: {error:.4f} m")

        if error < 1e-2:
            self.ik.move_to_configuration(q_soln)
        else:
            self.get_logger().warn("IK solution outside tolerance")

    def run_once(self):
        # Example target pose (x, y, z) in base_link with a simple yaw.
        target_point = [-0.043, -0.441, 0.654]
        target_rpy = [0.0, 0.0, -np.pi / 2]
        target_pose = self.ik.make_target_pose(target_point, target_rpy)

        q_init = self.ik.get_current_configuration()
        q_soln = self.ik.solve_pose_ik(target_pose, q_init=q_init)
        error = self.ik.compute_position_error(q_soln, target_point)
        self.get_logger().info(f"IK error: {error:.4f} m")

        if error < 1e-2:
            self.ik.move_to_configuration(q_soln)
        else:
            self.get_logger().warn("IK solution outside tolerance")


def main():
    node = IkExampleNode()
    node.run_once()


if __name__ == "__main__":
    main()
