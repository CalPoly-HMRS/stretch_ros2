#!/usr/bin/env python3

import argparse
from math import pi
import time

import rclpy

from hello_misc import HelloNode


def wait_for_joint_state(node, timeout_s=5.0):
	start = time.time()
	while time.time() - start < timeout_s:
		rclpy.spin_once(node, timeout_sec=0.1)
		if node.joint_state and node.joint_state.name:
			return True
	return False


def run_pose_test(steps=10, rate_hz=15.0):
	node = HelloNode.quick_create("pose_test", wait_for_first_pointcloud=False)
	if not wait_for_joint_state(node):
		node.get_logger().error("Timed out waiting for joint states. Aborting pose test.")
		rclpy.shutdown()
		return

	joint_values = {
		"lift": 0.6,
		"arm": 0.1,
		"wrist_pitch": 0.0,
		"wrist_roll": 0.0,
		"wrist_yaw": 0.0,
		"stretch_gripper": 0.0,
		"head_pan": 0.0,
		"head_tilt": 0.0,
		"base_translate": 0.0,
		"base_rotate": 0.0,
	}

	deltas = {
		"lift": 0.02,
		"arm": 0.02,
		"wrist_pitch": 0.1,
		"wrist_roll": 0.1,
		"wrist_yaw": 0.1,
		"stretch_gripper": 0.1,
		"head_pan": 0.1,
		"head_tilt": 0.1,
		"base_translate": 0.01,
		"base_rotate": 0.0,
	}

	period_s = 1.0 / rate_hz

	try:
		for _ in range(steps):
			joint_values = {k: v + deltas[k] for k, v in joint_values.items()}
			node.set_joint_poses(list(joint_values.items()))
			rclpy.spin_once(node, timeout_sec=0.1)
			time.sleep(period_s)
		time.sleep(5.0)  # hold the pose for a moment before moving back
		for _ in range(steps):
			joint_values = {k: v - deltas[k] for k, v in joint_values.items()}
			node.set_joint_poses(list(joint_values.items()))
			rclpy.spin_once(node, timeout_sec=0.1)
			time.sleep(period_s)
	finally:
		rclpy.shutdown()


def run_velocity_test(steps=40, rate_hz=20.0, duration_s=0.1):
	node = HelloNode.quick_create("velocity_test", wait_for_first_pointcloud=False)
	if not wait_for_joint_state(node):
		node.get_logger().error("Timed out waiting for joint states. Aborting velocity test.")
		rclpy.shutdown()
		return

	forward_vels = [
		("lift", 0.03),
		("arm", 0.03),
		("wrist_pitch", 0.05),
		("wrist_roll", 0.05),
		("wrist_yaw", 0.05),
		("head_pan", 0.05),
		("head_tilt", 0.05),
	]
	reverse_vels = [(name, -value) for name, value in forward_vels]

	period_s = 1.0 / rate_hz

	try:
		for _ in range(steps):
			node.set_joint_velocities(forward_vels, duration=duration_s)
			rclpy.spin_once(node, timeout_sec=0.1)
			time.sleep(period_s)

		for _ in range(steps):
			node.set_joint_velocities(reverse_vels, duration=duration_s)
			rclpy.spin_once(node, timeout_sec=0.1)
			time.sleep(period_s)
	finally:
		rclpy.shutdown()

def run_single_pose_test(hold_s=1.0):
	node = HelloNode.quick_create("single_pose_test", wait_for_first_pointcloud=False)
	if not wait_for_joint_state(node):
		node.get_logger().error("Timed out waiting for joint states. Aborting single pose test.")
		rclpy.shutdown()
		return

	start_pose = {
		"lift": 0.6,
		"arm": 0.1,
		"wrist_pitch": 0.0,
		"wrist_roll": 0.0,
		"wrist_yaw": 0.0,
		"stretch_gripper": 0.0,
		"head_pan": 0.0,
		"head_tilt": 0.0,
		"base_translate": 0.0,
		"base_rotate": 0.0,
	}

	middle_pose = {
		"lift": 0.4,
		"arm": 0.3,
		"wrist_pitch": 0.0,
		"wrist_roll": -pi / 4,
		"wrist_yaw": pi / 2,
		"stretch_gripper": 0.0,
		"head_pan": -pi / 2,
		"head_tilt": -pi / 4,
		"base_translate": 0.0,
		"base_rotate": 0.0,
	}

	try:
		node.set_joint_poses(list(start_pose.items()))
		rclpy.spin_once(node, timeout_sec=0.1)
		time.sleep(2.5)
		node.set_joint_poses(list(middle_pose.items()))
		rclpy.spin_once(node, timeout_sec=0.1)
		time.sleep(hold_s)
		node.set_joint_poses(list(start_pose.items()))
		rclpy.spin_once(node, timeout_sec=0.1)
		time.sleep(2.5)
	finally:
		rclpy.shutdown()


def parse_args():
	parser = argparse.ArgumentParser(description="Run pose or velocity tests.")
	subparsers = parser.add_subparsers(dest="command", required=True)

	pose_parser = subparsers.add_parser("pose", help="Run pose test")
	pose_parser.add_argument("--steps", type=int, default=10)
	pose_parser.add_argument("--rate", type=float, default=15.0)

	vel_parser = subparsers.add_parser("velocity", help="Run velocity test")
	vel_parser.add_argument("--steps", type=int, default=40)
	vel_parser.add_argument("--rate", type=float, default=20.0)
	vel_parser.add_argument("--duration", type=float, default=0.1)

	single_pose_parser = subparsers.add_parser("single_pose", help="Move to a pose, hold, then return")
	single_pose_parser.add_argument("--hold", type=float, default=2.5)

	return parser.parse_args()


def main():
	args = parse_args()
	if args.command == "pose":
		run_pose_test(steps=args.steps, rate_hz=args.rate)
	elif args.command == "velocity":
		run_velocity_test(steps=args.steps, rate_hz=args.rate, duration_s=args.duration)
	elif args.command == "single_pose":
		run_single_pose_test(hold_s=args.hold)


if __name__ == "__main__":
	main()
