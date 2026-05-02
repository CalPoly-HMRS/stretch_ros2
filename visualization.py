"""Visualization and display utilities."""

from __future__ import annotations

import cv2
import numpy as np


class Visualizer:
    """Handles frame annotation and display."""
    
    def __init__(self, window_name: str = "Full Live Tracking") -> None:
        """Initialize visualizer.
        
        Args:
            window_name: Name of the display window.
        """
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    def draw_detected_markers(
        self,
        image: np.ndarray,
        corners: list[np.ndarray] | None,
        ids: np.ndarray | None,
    ) -> None:
        """Draw detected ArUco markers on image.
        
        Args:
            image: Image to draw on (modified in place).
            corners: List of marker corner arrays.
            ids: Array of marker IDs.
        """
        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
    
    def draw_marker_axes(
        self,
        image: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        axis_length: float = 0.025,
    ) -> None:
        """Draw 3D coordinate axes on marker.
        
        Args:
            image: Image to draw on (modified in place).
            camera_matrix: Camera intrinsics matrix.
            dist_coeffs: Camera distortion coefficients.
            rvec: Rotation vector of marker.
            tvec: Translation vector of marker.
            axis_length: Length of axis lines in meters.
        """
        cv2.drawFrameAxes(
            image,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            axis_length,
        )
    
    def draw_angle_indicator(
        self,
        image: np.ndarray,
        angle_error_rad: float,
        error_visualization_length: int = 100,
    ) -> None:
        """Draw line indicating angle error at bottom center.
        
        Args:
            image: Image to draw on (modified in place).
            angle_error_rad: Angle error in radians.
            error_visualization_length: Length of indicator line in pixels.
        """
        h, w = image.shape[:2]
        start = (w // 2, h - 1)
        
        dx = int(error_visualization_length * np.sin(angle_error_rad))
        dy = int(error_visualization_length * np.cos(angle_error_rad))
        
        end_x = int(np.clip(start[0] + dx, 0, w - 1))
        end_y = int(np.clip(start[1] - dy, 0, h - 1))
        cv2.line(image, start, (end_x, end_y), (255, 0, 0), 4)
    
    def draw_text(
        self,
        image: np.ndarray,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] = (0, 255, 0),
        font_scale: float = 0.7,
        thickness: int = 2,
    ) -> None:
        """Draw text on image.
        
        Args:
            image: Image to draw on (modified in place).
            text: Text to display.
            position: (x, y) pixel position for text.
            color: BGR color tuple.
            font_scale: Font size scale.
            thickness: Text line thickness.
        """
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    
    def draw_hud(
        self,
        image: np.ndarray,
        fps: float,
        marker_count: int,
        status: str,
        selected_id: int | None = None,
        angle_error: float | None = None,
        wrist_yaw: float | None = None,
        tvec: np.ndarray | None = None,
        pre_guard_vel: float | None = None,
        post_guard_vel: float | None = None,
        show_fps: bool = True,
        show_marker_count: bool = True,
        show_status: bool = True,
        show_selected_id: bool = True,
        show_angle_error: bool = True,
        show_wrist_yaw: bool = True,
        show_tvec: bool = True,
        show_velocity_debug: bool = True,
    ) -> None:
        """Draw complete HUD (heads-up display) on image.
        
        Args:
            image: Image to draw on (modified in place).
            fps: Measured frames per second.
            marker_count: Number of detected markers.
            status: Status string ("TRACKING", "SEARCHING").
            selected_id: ID of selected marker, or None.
            angle_error: Angle error in radians, or None.
            tvec: Translation vector, or None.
        """
        y_pos = 28
        y_step = 28

        if show_fps:
            self.draw_text(image, f"Measured FPS: {fps:.1f}", (10, y_pos), (0, 255, 0))
            y_pos += y_step

        if show_marker_count:
            self.draw_text(image, f"Markers: {marker_count}", (10, y_pos), (0, 255, 255))
            y_pos += y_step

        if show_status:
            self.draw_text(image, f"Status: {status}", (10, y_pos), (255, 255, 0))
            y_pos += y_step

        if show_selected_id and selected_id is not None:
            self.draw_text(image, f"Selected ID: {selected_id}", (10, y_pos), (255, 200, 0))
            y_pos += y_step

        if show_angle_error and angle_error is not None:
            self.draw_text(
                image,
                f"Angle error (rad): {angle_error:+.3f}",
                (10, y_pos),
                (0, 200, 255),
            )
            y_pos += y_step

        if show_wrist_yaw and wrist_yaw is not None:
            self.draw_text(
                image,
                f"Wrist yaw (rad): {wrist_yaw:+.3f}",
                (10, y_pos),
                (0, 180, 180),
            )
            y_pos += y_step

        if show_tvec and tvec is not None:
            tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
            tvec_text = f"tvec (x,y,z): {tx:+.3f}, {ty:+.3f}, {tz:+.3f}"
            self.draw_text(image, tvec_text, (10, y_pos), (0, 180, 255))
            y_pos += y_step

        if show_velocity_debug and pre_guard_vel is not None and post_guard_vel is not None:
            self.draw_text(
                image,
                f"Vel pre/post: {pre_guard_vel:+.3f} / {post_guard_vel:+.3f}",
                (10, y_pos),
                (180, 180, 255),
            )
    
    def show_frame(self, image: np.ndarray) -> bool:
        """Display frame and check for quit signal.
        
        Args:
            image: Image to display.
        
        Returns:
            False if user pressed 'q', True otherwise.
        """
        cv2.imshow(self.window_name, image)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")
    
    def cleanup(self) -> None:
        """Clean up display resources."""
        cv2.destroyAllWindows()
