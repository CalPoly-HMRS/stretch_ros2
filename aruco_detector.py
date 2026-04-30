"""ArUco marker detection and pose estimation."""

from __future__ import annotations

from math import atan2

import cv2
import numpy as np
import pyrealsense2 as rs


def get_intrinsics_matrix_and_dist(
    color_profile: rs.video_stream_profile,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract camera matrix and distortion coefficients from RealSense profile.
    
    Args:
        color_profile: RealSense color stream profile.
    
    Returns:
        Tuple of (camera_matrix, distortion_coefficients).
    """
    intr = color_profile.get_intrinsics()
    camera_matrix = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.array(intr.coeffs, dtype=np.float64)
    return camera_matrix, dist_coeffs


def rotate_camera_matrix_90_clockwise(
    camera_matrix: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Rotate camera matrix 90 degrees clockwise for rotated camera feeds.
    
    Args:
        camera_matrix: Original camera intrinsics matrix.
        image_width: Width of the image after rotation.
        image_height: Height of the image after rotation.
    
    Returns:
        Rotated camera matrix.
    """
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    rotated_camera_matrix = np.array(
        [
            [fy, 0.0, (image_height - 1.0) - cy],
            [0.0, fx, cx],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rotated_camera_matrix


def estimate_single_marker_pose(
    selected_corners: np.ndarray,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate 3D pose of a single ArUco marker.
    
    Args:
        selected_corners: 2D corner coordinates of the marker (4x2).
        marker_size_m: Physical size of the marker in meters.
        camera_matrix: Camera intrinsics matrix.
        dist_coeffs: Camera distortion coefficients.
    
    Returns:
        Tuple of (rotation_vector, translation_vector) or (None, None) if estimation failed.
    """
    half = marker_size_m * 0.5
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    image_points = selected_corners.reshape(4, 2).astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None, None

    return rvec.reshape(3), tvec.reshape(3)


def compute_selected_marker_angle(
    corners: list[np.ndarray],
    ids: np.ndarray | None,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_size_m: float,
    target_tag_ids: tuple[int, ...],
) -> tuple[float | None, int | None, np.ndarray | None, np.ndarray | None]:
    """Compute angle error and pose of the selected ArUco marker.
    
    Selection priority: target_tag_ids in order, then first detected marker.
    
    Args:
        corners: List of detected marker corner arrays.
        ids: Array of detected marker IDs.
        camera_matrix: Camera intrinsics matrix (potentially rotated).
        dist_coeffs: Camera distortion coefficients.
        marker_size_m: Physical size of markers in meters.
        target_tag_ids: Tuple of IDs to prioritize, in order.
    
    Returns:
        Tuple of (angle_error_rad, selected_id, rotation_vector, translation_vector).
        Returns (None, None, None, None) if no markers detected or pose estimation failed.
    """
    if ids is None or len(ids) == 0:
        return None, None, None, None

    flat_ids = ids.flatten().tolist()

    selected_idx = None
    selected_id = None
    
    # Search for priority IDs
    if target_tag_ids:
        for wanted_id in target_tag_ids:
            for i, marker_id in enumerate(flat_ids):
                if marker_id == wanted_id:
                    selected_idx = i
                    selected_id = marker_id
                    break
            if selected_idx is not None:
                break
        if selected_idx is None:
            return None, None, None, None
    
    # Fall back to first detected marker
    if selected_idx is None:
        selected_idx = 0
        selected_id = flat_ids[0]

    selected_corners = np.array(corners[selected_idx], dtype=np.float32)
    rvec, tvec = estimate_single_marker_pose(
        selected_corners,
        marker_size_m,
        camera_matrix,
        dist_coeffs,
    )
    if tvec is None:
        return None, selected_id, None, None

    # Angle error is the horizontal (x) displacement relative to distance (z)
    angle_error = atan2(float(tvec[0]), float(tvec[2]))
    return angle_error, selected_id, rvec, tvec


class ArucoDetector:
    """Wrapper for ArUco marker detection."""
    
    def __init__(self, aruco_dict_id: int = cv2.aruco.DICT_6X6_250) -> None:
        """Initialize ArUco detector.
        
        Args:
            aruco_dict_id: OpenCV ArUco dictionary ID to use.
        """
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
    
    def detect_markers(self, gray_image: np.ndarray) -> tuple[list[np.ndarray] | None, np.ndarray | None]:
        """Detect ArUco markers in a grayscale image.
        
        Args:
            gray_image: Grayscale image to search for markers.
        
        Returns:
            Tuple of (corners_list, ids_array) or (None, None) if no markers found.
        """
        corners, ids, _ = self.detector.detectMarkers(gray_image)
        return corners, ids
