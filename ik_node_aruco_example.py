#!/usr/bin/env python3

# TODO: use the self.logger for every print...

import time

import numpy as np
import rclpy

from hello_misc import HelloNode, get_p1_to_p2_matrix
from ik_class import StretchIkRos

# First tag is the object to pick up, second tag is the target place to put it
TARGET_ARUCO_IDS = [0, 3]

GRIPPER_OPEN = 0.6
GRIPPER_CLOSED = 0.1


class IkArucoExampleNode(HelloNode):
    def __init__(self):
        super().__init__()
        self.main("ik_aruco_example_node", "ik_aruco_example_node", wait_for_first_pointcloud=False)
        self.ik = StretchIkRos(self, tool_name="tool_stretch_dex_wrist")

    def _wait_for_head_pan(self, target_pan, timeout_s=2.0, tolerance=0.03):
        end_time = time.time() + timeout_s
        while time.time() < end_time and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            rclpy.spin_once(self, timeout_sec=0.05)
            rclpy.spin_once(self, timeout_sec=0.05)
            rclpy.spin_once(self, timeout_sec=0.05)
            # run to get the latest joint state because im too lazy to multithread and its only one callback per spin
            if not self.joint_state or not self.joint_state.name:
                print("bro this should not be happening")
                continue
            current_pan = self.joint_state.position[self.joint_state.name.index("joint_head_pan")]
            if abs(current_pan - target_pan) <= tolerance:
                return True
        return False

    def _find_first_aruco_target(self, target_id, timeout=None, base_frame="base_link"):
        if target_id is None:
            self.get_logger().warn("target_id is None")
            return None, None
        tag_frame = f"aruco_tag_{target_id}"
        timeout_time = time.time() + timeout if timeout is not None else None
        while rclpy.ok() and (timeout_time is None or time.time() < timeout_time):
            # time.sleep(2) # Wait 2 seconds before trying again, just cuz i felt like it
            tag_to_base, _ = get_p1_to_p2_matrix(
                tag_frame,
                base_frame,
                self.tf2_buffer,
                timeout_s=0.25,
                verbose=False,
            )
            if tag_to_base is None:
                rclpy.spin_once(self, timeout_sec=0.1)
                continue
            target_point = (tag_to_base @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
            return target_id, target_point
        return None, None


    def pan_and_search(self, target_id, timeout=None):
        timeout_time = time.time() + timeout if timeout is not None else None
        head_pan_angles = np.linspace(np.pi / 4, -3 * np.pi / 4, num=10)
        tag_frame = f"aruco_tag_{target_id}"
        while rclpy.ok() and (timeout_time is None or time.time() < timeout_time):
            for head_pan in head_pan_angles:
                self.set_joint_poses([("head_pan", head_pan)])
                rclpy.spin_once(self, timeout_sec=0.5)
                self._wait_for_head_pan(head_pan)
                tag_to_base, _ = get_p1_to_p2_matrix(
                    tag_frame,
                    "base_link",
                    self.tf2_buffer,
                    timeout_s=0.25,
                    verbose=False,
                )
                if tag_to_base is None:
                    if time.time() >= timeout_time:
                        self.get_logger().warn(f"Timeout reached while searching for aruco_tag_{target_id}")
                        return None, None
                    rclpy.spin_once(self, timeout_sec=0.1)
                    rclpy.spin_once(self, timeout_sec=0.1)
                    rclpy.spin_once(self, timeout_sec=0.1)
                    time.sleep(0.5)
                    rclpy.spin_once(self, timeout_sec=0.1)
                    continue
                target_point = (tag_to_base @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
                return target_id, target_point
        return None, None

    def run_once(self):
        self.set_joint_poses([("head_tilt", -np.pi / 4)])

        answer = input("Open gripper AND move wrist? [y/N]: ").strip().lower()
        if answer.startswith("y"):
            # self.set_joint_poses([("stretch_gripper", GRIPPER_OPEN)])
            # self.set_joint_poses([("wrist_pitch", -0.8)])
            # self.set_joint_poses([("wrist_roll", 0.0)])
            self.set_joint_poses([("stretch_gripper", GRIPPER_OPEN), ("wrist_pitch", -0.8), ("wrist_roll", 0.0)])
            time.sleep(0.5)
        else:
            print("returning early")
            return
        
        print("Searching for first aruco target...")

        marker_id, target_point = self.pan_and_search(target_id=TARGET_ARUCO_IDS[0], timeout=30.0)

        print(f"Pan and search result: marker_id={marker_id}, target_point={target_point}")
        time.sleep(1.0)

        marker_id, target_point = self._find_first_aruco_target(target_id=TARGET_ARUCO_IDS[0])
        if target_point is None:
            # self.get_logger().warn("No aruco_tag_* frame found in TF.")
            return

        q_init = self.ik.get_current_configuration(tool_name="tool_stretch_dex_wrist")

        # print target point
        print(f"Target point: {target_point}")

        # add a little z offset to avoid colliding with the table
        target_point[2] += 0.1

        # print target point again
        print(f"Target point with z offset: {target_point}")

        # print current wrist pitch
        current_wrist_pitch = self.ik._get_q_value(q_init, "joint_wrist_pitch")
        print(f"Current wrist pitch (q_init): {current_wrist_pitch:.3f}")

        current_wrist_pitch = self.ik._joint_pos("joint_wrist_pitch")
        print(f"Current wrist pitch: {current_wrist_pitch:.3f}")

        answer = input(f"Calculate ik for aruco_tag_{marker_id}? [y/N]: ").strip().lower()
        if not answer.startswith("y"):
            self.get_logger().info("Exiting.")
            return

        # set bounds for the wrist pitch to approach from the top and solve ik
        q_soln = self.ik.solve_point_ik(
            target_point,
            q_init=q_init,
            joint_bounds={"wrist_pitch": (-1.2, -0.5)},
            fixed_joints=["base_translate", "wrist_roll"],
        )
        error = self.ik.compute_position_error(q_soln, target_point)
        self.get_logger().info(f"IK error: {error:.4f} m")

        answer = input(f"Pre-move lift to {self.ik._get_q_value(q_soln, 'joint_lift')} ? [y/N]: ").strip().lower()
        if not answer.startswith("y"):
            self.get_logger().info("Skipping move.")
            return
        
        # pre move lift to avoid hitting table
        self.set_joint_poses([("lift", self.ik._get_q_value(q_soln, "joint_lift"))])

        answer = input(f"Move to aruco_tag_{marker_id}? [y/N]: ").strip().lower()
        if not answer.startswith("y"):
            self.get_logger().info("Skipping move.")
            return

        # move for pickup
        if error < 0.5:
            self.ik.move_to_configuration(q_soln, tool_name="tool_stretch_dex_wrist")

            answer = input("Move lift down a bit? [y/N]: ").strip().lower()
            if answer.startswith("y"):
                self.set_joint_poses([("lift", self.ik._get_q_value(q_soln, 'joint_lift') - 0.1)])

            answer = input("Close gripper? [y/N]: ").strip().lower()
            if answer.startswith("y"):
                self.set_joint_poses([("stretch_gripper", GRIPPER_CLOSED)])

            answer = input("Move lift up a bit? [y/N]: ").strip().lower()
            if answer.startswith("y"):
                self.set_joint_poses([("lift", self.ik._get_q_value(q_soln, 'joint_lift'))])
        else:
            self.get_logger().warn("IK solution outside tolerance")

        answer = input(f"start search for next aruco tag? [y/N]: ").strip().lower()
        if not answer.startswith("y"):
            self.get_logger().info("Skipping move.")
            return

        marker_id, next_target_point = self.pan_and_search(target_id=TARGET_ARUCO_IDS[1], timeout=30.0)

        print(f"Pan and search result: marker_id={marker_id}, target_point={next_target_point}")
        time.sleep(1.0)

        next_marker_id, next_target_point = self._find_first_aruco_target(target_id=TARGET_ARUCO_IDS[1])

        answer = input(f"Place object on next aruco tag? [y/N]: ").strip().lower()
        if not answer.startswith("y") or next_target_point is None:
            self.get_logger().info("Smth went wrong, exiting.")
            return

        print(f"Next target point: {next_target_point}")
        next_target_point[2] += 0.1
        print(f"Next target point with z offset: {next_target_point}")

        answer = input(f"Calculate ik for aruco_tag_{next_marker_id}? [y/N]: ").strip().lower()
        if not answer.startswith("y"):
            self.get_logger().info("Exiting.")
            return

        q_init = self.ik.get_current_configuration(tool_name="tool_stretch_dex_wrist")
        q_soln = self.ik.solve_point_ik(
            next_target_point,
            q_init=q_init,
            joint_bounds={"wrist_pitch": (-1.2, -0.5)},
            fixed_joints=["base_translate", "wrist_roll"],
        )
        error = self.ik.compute_position_error(q_soln, next_target_point)
        self.get_logger().info(f"IK error: {error:.4f} m")

        answer = input(f"Pre-rotate base? [y/N]: ").strip().lower()
        if answer.startswith("y"):
            base_rotate_soln = self.ik._get_q_value(q_soln, "base_rotate")
            print(f"Base rotate solution: {base_rotate_soln:.3f} rad")
            self.set_joint_poses([("base_rotate", self.ik._get_q_value(q_soln, "base_rotate"))])

        # recalculate ik after rotating base because otherwise itll move the base again
        q_init = self.ik.get_current_configuration(tool_name="tool_stretch_dex_wrist")
        q_soln = self.ik.solve_point_ik(
            next_target_point,
            q_init=q_init,
            joint_bounds={"wrist_pitch": (-1.2, -0.5)},
            fixed_joints=["base_translate", "base_rotate", "wrist_roll"],
        )

        answer = input(f"Move to aruco_tag_{next_marker_id}? [y/N]: ").strip().lower()
        if not answer.startswith("y"):
            self.get_logger().info("Skipping move.")
            return

        if error < 0.5:
            self.ik.move_to_configuration(q_soln, tool_name="tool_stretch_dex_wrist")

            answer = input("Open gripper? [y/N]: ").strip().lower()
            if answer.startswith("y"):
                self.set_joint_poses([("stretch_gripper", GRIPPER_OPEN)])
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