"""TF publishing utilities for ArUco tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from rclpy.node import Node

try:
    from geometry_msgs.msg import TransformStamped
    from rclpy.duration import Duration
    from tf2_ros import Buffer, TransformBroadcaster, TransformListener
    from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
except ImportError:  # pragma: no cover - ROS2 may not be installed on dev machines
    TransformStamped = None
    TransformBroadcaster = None
    Buffer = None
    TransformListener = None
    Duration = None
    ConnectivityException = None
    ExtrapolationException = None
    LookupException = None


@dataclass(frozen=True)
class TfPublisherConfig:
    """Configuration for TF publishing."""

    base_frame_id: str
    camera_frame_id: str
    marker_frame_prefix: str
    publish_base_to_camera_identity: bool


def _rotation_matrix_to_quaternion(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a quaternion (x, y, z, w)."""
    m00, m01, m02 = rot[0, 0], rot[0, 1], rot[0, 2]
    m10, m11, m12 = rot[1, 0], rot[1, 1], rot[1, 2]
    m20, m21, m22 = rot[2, 0], rot[2, 1], rot[2, 2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return float(x), float(y), float(z), float(w)


def _rvec_to_quaternion(rvec: np.ndarray) -> tuple[float, float, float, float]:
    """Convert an OpenCV rotation vector to a quaternion (x, y, z, w)."""
    rot, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return _rotation_matrix_to_quaternion(rot)


class TfPublisher:
    """Publishes TF frames for base, camera, and markers."""

    def __init__(self, node: "Node", config: TfPublisherConfig) -> None:
        if TransformBroadcaster is None or TransformStamped is None:
            raise RuntimeError("ROS2 tf2_ros is not available; install it to publish TF.")
        self.node = node
        self.config = config
        self.broadcaster = TransformBroadcaster(node)

    def publish_base_to_camera_identity(self, stamp) -> None:
        """Publish a static identity transform from base to camera."""
        if not self.config.publish_base_to_camera_identity:
            return

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.config.base_frame_id
        transform.child_frame_id = self.config.camera_frame_id
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        self.broadcaster.sendTransform(transform)


class TfLookupHelper:
    """Helper for querying TF transforms."""

    def __init__(self, node: "Node") -> None:
        if Buffer is None or TransformListener is None or Duration is None:
            raise RuntimeError("ROS2 tf2_ros is not available; install it to query TF.")
        self.node = node
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)

    def lookup_transform(
        self,
        target_frame: str,
        source_frame: str,
        timeout_s: float = 0.1,
    ) -> TransformStamped | None:
        """Look up the transform that maps source_frame into target_frame."""
        try:
            return self.buffer.lookup_transform(
                target_frame,
                source_frame,
                self.node.get_clock().now(),
                timeout=Duration(seconds=float(timeout_s)),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def publish_marker(self, marker_id: int, rvec: np.ndarray, tvec: np.ndarray, stamp) -> None:
        """Publish camera->marker transform for the given ArUco tag."""
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.config.camera_frame_id
        transform.child_frame_id = f"{self.config.marker_frame_prefix}{marker_id}"

        transform.transform.translation.x = float(tvec[0])
        transform.transform.translation.y = float(tvec[1])
        transform.transform.translation.z = float(tvec[2])

        qx, qy, qz, qw = _rvec_to_quaternion(rvec)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw

        self.broadcaster.sendTransform(transform)
