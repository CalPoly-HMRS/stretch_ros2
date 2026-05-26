"""ArUco marker detection and pose estimation."""

from __future__ import annotations

from math import atan2

import cv2
import numpy as np
import pyrealsense2 as rs

import hello_helpers.fit_plane as fp


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


def _normalize_vector(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return None
    return vec / norm


def _compute_depth_points_in_polygon(
    depth_image_m: np.ndarray,
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, float | None]:
    height, width = depth_image_m.shape[:2]
    poly = np.round(corners.reshape(-1, 2)).astype(np.int32)

    x_min = max(0, int(np.min(poly[:, 0])))
    x_max = min(width - 1, int(np.max(poly[:, 0])))
    y_min = max(0, int(np.min(poly[:, 1])))
    y_max = min(height - 1, int(np.max(poly[:, 1])))

    if x_max <= x_min or y_max <= y_min:
        return np.empty((0, 3), dtype=np.float64), None

    depth_crop = depth_image_m[y_min : y_max + 1, x_min : x_max + 1]
    if depth_crop.size == 0:
        return np.empty((0, 3), dtype=np.float64), None

    mask = np.zeros(depth_crop.shape, dtype=np.uint8)
    poly_shift = poly - [x_min, y_min]
    cv2.fillConvexPoly(mask, poly_shift, 255)

    coords = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    ys = coords[0]
    xs = coords[1]

    z = depth_crop
    valid = (
        (mask > 0)
        & np.isfinite(z)
        & (z > min_depth_m)
        & (z < max_depth_m)
    )

    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), None

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    x = ((xs - cx) / fx) * z
    y = ((ys - cy) / fy) * z

    points = np.stack([x, y, z], axis=-1)[valid]
    median_depth = float(np.median(z[valid]))
    return points.astype(np.float64), median_depth


def _fit_plane_pose_from_depth(
    depth_points: np.ndarray,
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    rvec_fallback: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    plane = fp.FitPlane()
    plane.fit_svd(depth_points, verbose=False)
    if plane.n is None or plane.d is None:
        return None, None

    n = np.reshape(plane.n, (3, 1))
    d = float(plane.d)

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    def pix_to_plane(pix_x: float, pix_y: float) -> np.ndarray | None:
        ray = np.array([(pix_x - cx) / fx, (pix_y - cy) / fy, 1.0], dtype=np.float64)
        ray_norm = _normalize_vector(ray)
        if ray_norm is None:
            return None
        denom = float(n.T @ ray_norm.reshape(3, 1))
        if abs(denom) < 1e-6:
            return None
        return (d / denom) * ray_norm

    corner_points = []
    for (pix_x, pix_y) in corners.reshape(4, 2):
        point = pix_to_plane(float(pix_x), float(pix_y))
        if point is None:
            return None, None
        corner_points.append(point)

    marker_position = np.mean(corner_points, axis=0)

    top_left, top_right, bottom_right, bottom_left = corner_points
    y_axis = (top_left + top_right) - (bottom_left + bottom_right)
    x_axis = (top_right + bottom_right) - (top_left + bottom_left)

    y_axis = _normalize_vector(y_axis)
    x_axis = _normalize_vector(x_axis)

    rotation_matrix, _ = cv2.Rodrigues(rvec_fallback)
    old_x_axis = x_axis if x_axis is not None else rotation_matrix[:, 0]
    old_y_axis = y_axis if y_axis is not None else rotation_matrix[:, 1]

    new_z_axis = _normalize_vector(plane.get_plane_normal().reshape(3))
    if new_z_axis is None:
        return None, None

    if x_axis is not None and y_axis is None:
        new_x_axis = _normalize_vector(old_x_axis - (new_z_axis * np.dot(new_z_axis, old_x_axis)))
        if new_x_axis is None:
            return None, None
        new_y_axis = np.cross(new_z_axis, new_x_axis)
    elif x_axis is None and y_axis is not None:
        new_y_axis = _normalize_vector(old_y_axis - (new_z_axis * np.dot(new_z_axis, old_y_axis)))
        if new_y_axis is None:
            return None, None
        new_x_axis = np.cross(new_y_axis, new_z_axis)
    else:
        new_y_axis_1 = _normalize_vector(old_y_axis - (new_z_axis * np.dot(new_z_axis, old_y_axis)))
        new_x_axis_2 = _normalize_vector(old_x_axis - (new_z_axis * np.dot(new_z_axis, old_x_axis)))
        if new_y_axis_1 is None or new_x_axis_2 is None:
            return None, None
        new_x_axis_1 = np.cross(new_y_axis_1, new_z_axis)
        new_x_axis = _normalize_vector((new_x_axis_1 + new_x_axis_2) * 0.5)
        if new_x_axis is None:
            return None, None
        new_y_axis = np.cross(new_z_axis, new_x_axis)

    new_x_axis = _normalize_vector(new_x_axis)
    new_y_axis = _normalize_vector(new_y_axis)
    if new_x_axis is None or new_y_axis is None:
        return None, None

    rotation = np.column_stack([new_x_axis, new_y_axis, new_z_axis])
    refined_rvec, _ = cv2.Rodrigues(rotation)
    return refined_rvec.reshape(3), marker_position.reshape(3)


def refine_pose_with_depth(
    corners: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    depth_image_m: np.ndarray | None,
    mode: str,
    min_depth_m: float,
    max_depth_m: float,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if depth_image_m is None or mode == "off":
        return rvec, tvec, False

    points, median_depth = _compute_depth_points_in_polygon(
        depth_image_m,
        corners,
        camera_matrix,
        min_depth_m,
        max_depth_m,
    )

    if points.shape[0] < min_points or median_depth is None:
        return rvec, tvec, False

    if mode == "simple":
        # Scale the translation vector so that its z component matches the median depth.
        if tvec is None or tvec[2] <= 0.0:
            return rvec, tvec, False
        scale = median_depth / float(tvec[2])
        return rvec, (tvec * scale).reshape(3), True

    if mode == "plane":
        # Fit a plane to the depth points and re-estimate the pose based on that plane.
        # this is how the stretch one does it
        refined_rvec, refined_tvec = _fit_plane_pose_from_depth(
            points,
            corners,
            camera_matrix,
            rvec,
        )
        if refined_rvec is None or refined_tvec is None:
            return rvec, tvec, False
        return refined_rvec, refined_tvec, True

    return rvec, tvec, False


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