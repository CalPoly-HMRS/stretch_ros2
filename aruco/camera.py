"""RealSense camera management and frame capture."""

from __future__ import annotations

import numpy as np
import pyrealsense2 as rs


class CameraManager:
    """Manages RealSense camera pipeline and frame acquisition."""
    
    def __init__(
        self,
        device_index: int = 1,
        profile_index: int = 0,
        depth_profile_index: int | None = None,
        enable_depth: bool = True,
        align_depth_to_color: bool = True,
    ) -> None:
        """Initialize camera manager.
        
        Args:
            device_index: Index of RealSense device to use (0, 1, etc.).
            profile_index: Index into the sorted list of color stream profiles.
            depth_profile_index: Index into the sorted list of depth stream profiles.
            enable_depth: Whether to enable depth streaming.
            align_depth_to_color: Whether to align depth frames to the color stream.
        """
        self.device_index = device_index
        self.profile_index = profile_index
        self.depth_profile_index = depth_profile_index
        self.enable_depth = enable_depth
        self.align_depth_to_color = align_depth_to_color
        self.target_width = 0
        self.target_height = 0
        self.target_fps = 0
        self.target_fmt = rs.format.bgr8
        self.depth_width = 0
        self.depth_height = 0
        self.depth_fps = 0
        self.depth_fmt = rs.format.z16
        
        self.pipeline: rs.pipeline | None = None
        self.selected_serial: str | None = None
        self.color_frame_profile: rs.video_stream_profile | None = None
        self.depth_frame_profile: rs.video_stream_profile | None = None
        self.depth_scale: float | None = None
        self.align: rs.align | None = None

    def _select_depth_profile(
        self,
        profiles: list[tuple[int, int, int, rs.format]],
        target_width: int,
        target_height: int,
        target_fps: int,
    ) -> tuple[int, int, int, rs.format] | None:
        if not profiles:
            return None

        if self.depth_profile_index is not None and self.depth_profile_index >= 0:
            if self.depth_profile_index >= len(profiles):
                return None
            return profiles[self.depth_profile_index]

        z16_profiles = [profile for profile in profiles if profile[3] == rs.format.z16]
        candidates = z16_profiles if z16_profiles else profiles

        def score(profile: tuple[int, int, int, rs.format]) -> tuple[int, int, int]:
            width, height, fps, _ = profile
            area_diff = abs((width * height) - (target_width * target_height))
            fps_diff = abs(fps - target_fps)
            exact_match = 0 if (width == target_width and height == target_height) else 1
            return (exact_match, area_diff, fps_diff)

        return sorted(candidates, key=score)[0]
    
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

        profiles: set[tuple[int, int, int, rs.format]] = set()
        for sensor in device.sensors:
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() != rs.stream.color:
                    continue
                try:
                    fmt = profile.format()
                    vprofile = profile.as_video_stream_profile()
                    profiles.add((vprofile.width(), vprofile.height(), vprofile.fps(), fmt))
                except RuntimeError:
                    continue

        sorted_profiles = sorted(profiles, key=lambda x: (x[0] * x[1], x[2], str(x[3])))
        if not sorted_profiles:
            print("No color stream profiles found for the selected device.")
            return False

        if self.profile_index < 0 or self.profile_index >= len(sorted_profiles):
            print(
                f"Profile index {self.profile_index} out of range. Found {len(sorted_profiles)} profiles."
            )
            return False

        self.target_width, self.target_height, self.target_fps, self.target_fmt = sorted_profiles[
            self.profile_index
        ]
        
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

        if self.enable_depth:
            depth_profiles: set[tuple[int, int, int, rs.format]] = set()
            for sensor in device.sensors:
                for profile in sensor.get_stream_profiles():
                    if profile.stream_type() != rs.stream.depth:
                        continue
                    try:
                        fmt = profile.format()
                        vprofile = profile.as_video_stream_profile()
                        depth_profiles.add((vprofile.width(), vprofile.height(), vprofile.fps(), fmt))
                    except RuntimeError:
                        continue

            depth_profile = self._select_depth_profile(
                sorted(depth_profiles),
                self.target_width,
                self.target_height,
                self.target_fps,
            )
            if depth_profile is None:
                print("No depth stream profiles found for the selected device.")
                return False

            self.depth_width, self.depth_height, self.depth_fps, self.depth_fmt = depth_profile
            config.enable_stream(
                rs.stream.depth,
                self.depth_width,
                self.depth_height,
                self.depth_fmt,
                self.depth_fps,
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
        if self.enable_depth:
            self.depth_frame_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())
            if self.align_depth_to_color:
                self.align = rs.align(rs.stream.color)
            print(
                "Camera initialized: "
                f"color={self.target_width}x{self.target_height}@{self.target_fps}, "
                f"depth={self.depth_width}x{self.depth_height}@{self.depth_fps}"
            )
        else:
            print(f"Camera initialized: {self.target_width}x{self.target_height}@{self.target_fps}")
        return True
    
    def get_frames(self, timeout_ms: int = 1500) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Capture a single color frame and optional depth frame.
        
        Args:
            timeout_ms: Timeout in milliseconds for waiting on frame.
        
        Returns:
            Tuple of (BGR image, depth image in meters) or (None, None) if unavailable.
        """
        if self.pipeline is None:
            return None, None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError:
            return None, None

        if self.enable_depth and self.align is not None:
            frames = self.align.process(frames)
        
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None, None
        
        image = np.asanyarray(color_frame.get_data())
        depth_image = None

        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame and self.depth_scale is not None:
                depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                depth_image = depth_raw * self.depth_scale

        return image, depth_image

    def get_frame(self, timeout_ms: int = 1500) -> np.ndarray | None:
        """Capture a single color frame (backwards-compatible helper)."""
        color, _ = self.get_frames(timeout_ms=timeout_ms)
        return color
    
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