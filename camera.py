"""RealSense camera management and frame capture."""

from __future__ import annotations

import numpy as np
import pyrealsense2 as rs


class CameraManager:
    """Manages RealSense camera pipeline and frame acquisition."""
    
    def __init__(
        self,
        device_index: int = 1,
        target_width: int = 424,
        target_height: int = 240,
        target_fps: int = 60,
        fmt: rs.format = rs.format.bgr8,
    ) -> None:
        """Initialize camera manager.
        
        Args:
            device_index: Index of RealSense device to use (0, 1, etc.).
            target_width: Desired image width in pixels.
            target_height: Desired image height in pixels.
            target_fps: Desired frame rate in Hz.
            fmt: OpenCV format for frames (rs.format.bgr8, etc.).
        """
        self.device_index = device_index
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.target_fmt = fmt
        
        self.pipeline: rs.pipeline | None = None
        self.selected_serial: str | None = None
        self.color_frame_profile: rs.video_stream_profile | None = None
    
    def initialize(self) -> bool:
        """Initialize and start the RealSense pipeline.
        
        Returns:
            True if successful, False otherwise.
        """
        ctx = rs.context()
        devices = list(ctx.query_devices())
        
        if not devices:
            print("No RealSense devices detected.")
            return False
        
        if self.device_index >= len(devices):
            print(f"Device index {self.device_index} out of range. Found {len(devices)} devices.")
            return False
        
        device = devices[self.device_index]
        self.selected_serial = device.get_info(rs.camera_info.serial_number)
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.selected_serial)
        config.enable_stream(
            rs.stream.color,
            self.target_width,
            self.target_height,
            self.target_fmt,
            self.target_fps,
        )
        
        try:
            profile = self.pipeline.start(config)
        except RuntimeError as exc:
            print(
                f"Failed to start stream at {self.target_width}x{self.target_height}"
                f"@{self.target_fps}, format={self.target_fmt}: {exc}"
            )
            return False
        
        self.color_frame_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        print(f"Camera initialized: {self.target_width}x{self.target_height}@{self.target_fps}")
        return True
    
    def get_frame(self, timeout_ms: int = 1500) -> np.ndarray | None:
        """Capture a single color frame.
        
        Args:
            timeout_ms: Timeout in milliseconds for waiting on frame.
        
        Returns:
            BGR image as numpy array, or None if frame not available.
        """
        if self.pipeline is None:
            return None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError:
            return None
        
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        
        image = np.asanyarray(color_frame.get_data())
        return image
    
    def get_intrinsics(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get camera matrix and distortion coefficients.
        
        Returns:
            Tuple of (camera_matrix, distortion_coeffs) or None if not initialized.
        """
        if self.color_frame_profile is None:
            return None
        
        from aruco_detector import get_intrinsics_matrix_and_dist
        return get_intrinsics_matrix_and_dist(self.color_frame_profile)
    
    def stop(self) -> None:
        """Stop the camera pipeline."""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
