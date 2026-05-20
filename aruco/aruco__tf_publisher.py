#!/usr/bin/env python3
"""Publish ArUco tag poses to ROS 2 TF."""

from __future__ import annotations

import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_matrix


_OPTICAL_TO_LINK_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
class ArucoTfPublisher:
    """Publish marker poses to TF using a ROS 2 node."""

    def __init__(
        self,
        node: Node,
        parent_frame: str = "camera_color_optical_frame",
        child_frame_prefix: str = "aruco_tag_",
    ) -> None:
        self.node = node
        self.parent_frame = parent_frame
        self.child_frame_prefix = child_frame_prefix
        self.tf_broadcaster = TransformBroadcaster(node)

    def publish(self, rvec: np.ndarray, tvec: np.ndarray, marker_id: int) -> None:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        rotation_matrix = _OPTICAL_TO_LINK_ROTATION @ rotation_matrix
        translated = _OPTICAL_TO_LINK_ROTATION @ tvec.reshape(3)
        transform_matrix = np.eye(4, dtype=np.float64)
        transform_matrix[:3, :3] = rotation_matrix
        quat = quaternion_from_matrix(transform_matrix)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.node.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.parent_frame
        tf_msg.child_frame_id = f"{self.child_frame_prefix}{marker_id}"
        tf_msg.transform.translation.x = float(translated[0])
        tf_msg.transform.translation.y = float(translated[1])
        tf_msg.transform.translation.z = float(translated[2])
        tf_msg.transform.rotation.x = float(quat[0])
        tf_msg.transform.rotation.y = float(quat[1])
        tf_msg.transform.rotation.z = float(quat[2])
        tf_msg.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(tf_msg)
