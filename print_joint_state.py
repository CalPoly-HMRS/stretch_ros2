#!/usr/bin/env python3

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


def main():
    node = HelloNode.quick_create("print_joint_state", wait_for_first_pointcloud=False)
    try:
        if not wait_for_joint_state(node):
            node.get_logger().error("Timed out waiting for joint states.")
            return
        rclpy.spin_once(node, timeout_sec=0.1)
        print("JointState names:")
        for name in node.joint_state.name:
            print(f"- {name}")
        print("\nJointState positions:")
        for name, pos in zip(node.joint_state.name, node.joint_state.position):
            print(f"- {name}: {pos}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
