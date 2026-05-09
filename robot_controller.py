"""Stretch robot motion control."""

from __future__ import annotations

import time
from math import pi

import numpy as np

try:
    import stretch_body.robot as stretch_robot
except ImportError:
    stretch_robot = None


class StretchWristTracker:
    """Controls Stretch robot wrist for visual tracking."""
    
    def __init__(
        self,
        command_hz: float = 20.0,
        camera_follows_wrist: bool = False,
        wrist_direction_sign: float = -1.0,
        deadband_rad: float = 0.012,
        wrist_yaw_min_rad: float = -1.39,
        wrist_yaw_max_rad: float = 4.42,
        wrist_yaw_limit_buffer_rad: float = 0.0,
        control_kp: float = 2.0,
        max_wrist_speed_rad_s: float = 1.8,
        max_wrist_accel_rad_s2: float = 6.0,
        error_smoothing_alpha: float = 0.25,
    ) -> None:
        """Initialize wrist tracker.
        
        Args:
            command_hz: Command update rate in Hz.
            camera_follows_wrist: Whether camera follows wrist motion.
            wrist_direction_sign: Direction multiplier for wrist rotation (+1 or -1).
            deadband_rad: Angle error threshold below which wrist stops moving.
            wrist_yaw_min_rad: Minimum wrist yaw angle in radians.
            wrist_yaw_max_rad: Maximum wrist yaw angle in radians.
            wrist_yaw_limit_buffer_rad: Buffer before yaw limits to stop motion.
            control_kp: Proportional control gain.
            max_wrist_speed_rad_s: Maximum wrist velocity in rad/s.
            max_wrist_accel_rad_s2: Maximum wrist acceleration in rad/s².
            error_smoothing_alpha: Exponential smoothing factor (0-1).
        """
        if stretch_robot is None:
            raise RuntimeError(
                "stretch_body is not installed. Install it on the robot before running this script."
            )
        
        self.robot = stretch_robot.Robot()
        self.camera_follows_wrist = camera_follows_wrist
        
        self.wrist_deadband_rad = deadband_rad
        self.wrist_direction_sign = wrist_direction_sign
        self.wrist_yaw_min_rad = wrist_yaw_min_rad
        self.wrist_yaw_max_rad = wrist_yaw_max_rad
        self.wrist_yaw_limit_buffer_rad = max(0.0, wrist_yaw_limit_buffer_rad)
        self.control_kp = control_kp
        self.max_wrist_speed_rad_s = max_wrist_speed_rad_s
        self.max_wrist_accel_rad_s2 = max_wrist_accel_rad_s2
        self.error_smoothing_alpha = error_smoothing_alpha
        
        self.min_command_interval_s = 1.0 / command_hz
        self._last_command_time_s = 0.0
        self._smoothed_error_rad = 0.0
        self._last_cmd_vel = 0.0
        self._last_seen_time_s = 0.0
        self._last_pre_guard_vel = 0.0
        self._last_post_guard_vel = 0.0
    
    def clamp_wrist_yaw_error(self, current_yaw: float, error_rad: float) -> float:
        """Clamp yaw error so target stays within hard joint limits.
        
        Args:
            current_yaw: Current wrist yaw angle in radians.
            error_rad: Desired yaw error in radians.
        
        Returns:
            Clamped yaw error in radians.
        """
        min_limit = self.wrist_yaw_min_rad
        max_limit = self.wrist_yaw_max_rad
        if min_limit >= max_limit:
            return 0.0

        target_yaw = float(np.clip(current_yaw + error_rad, min_limit, max_limit))
        return target_yaw - current_yaw

    def _apply_yaw_limit_velocity_guard(
        self,
        current_yaw: float,
        desired_vel: float,
    ) -> float:
        """Soft-limit velocity near yaw bounds while allowing recovery."""
        hard_min = self.wrist_yaw_min_rad
        hard_max = self.wrist_yaw_max_rad
        if hard_min >= hard_max:
            return 0.0

        buffer_rad = max(0.0, self.wrist_yaw_limit_buffer_rad)
        soft_min = hard_min + buffer_rad
        soft_max = hard_max - buffer_rad

        if current_yaw <= hard_min and desired_vel < 0.0:
            return 0.0
        if current_yaw >= hard_max and desired_vel > 0.0:
            return 0.0

        if buffer_rad > 0.0:
            if desired_vel < 0.0 and current_yaw < soft_min:
                scale = (current_yaw - hard_min) / buffer_rad
                return desired_vel * float(np.clip(scale, 0.0, 1.0))
            if desired_vel > 0.0 and current_yaw > soft_max:
                scale = (hard_max - current_yaw) / buffer_rad
                return desired_vel * float(np.clip(scale, 0.0, 1.0))

        return desired_vel
    
    def initialize_robot(self) -> None:
        """Connect to robot hardware and initialize pose.
        
        Raises:
            RuntimeError: If robot connection fails.
        """
        did_startup = self.robot.startup()
        if not did_startup:
            raise RuntimeError("Failed to connect to Stretch hardware via stretch_body.robot.Robot().")
        
        print("Connected to Stretch hardware.")
        if not self.robot.is_homed():
            print("Warning: Robot is not homed. Tracking may be inaccurate until homed.")
        
        # Move to neutral tracking pose
        try:
            self.robot.head.move_to("head_pan", -pi / 2)
            self.robot.head.move_to("head_tilt", -pi / 8)
        except Exception:
            pass
        
        if "wrist_yaw" in self.robot.end_of_arm.joints:
            self.robot.end_of_arm.move_to("wrist_yaw", 0.0)
        
        if "wrist_pitch" in self.robot.end_of_arm.joints:
            self.robot.end_of_arm.move_to("wrist_pitch", 0.0)
        
        if "wrist_roll" in self.robot.end_of_arm.joints:
            self.robot.end_of_arm.move_to("wrist_roll", 0.0)
        
        print("Robot initialized. Stretch-body wrist tracking is active.")
    
    def _send_wrist_velocity(self, cmd_vel: float) -> None:
        """Send velocity command to wrist yaw joint.
        
        Args:
            cmd_vel: Velocity in rad/s.
        """
        try:
            self.robot.end_of_arm.get_joint("wrist_yaw").set_velocity(float(cmd_vel))
        except Exception:
            return
    
    def _smooth_error(self, raw_error_rad: float) -> float:
        """Apply exponential smoothing to angle error.
        
        Args:
            raw_error_rad: Unsmoothed angle error in radians.
        
        Returns:
            Smoothed angle error in radians.
        """
        alpha = self.error_smoothing_alpha
        if raw_error_rad * self._smoothed_error_rad < 0.0:
            self._smoothed_error_rad = 0.0
        self._smoothed_error_rad = (
            (1.0 - alpha) * self._smoothed_error_rad + alpha * raw_error_rad
        )
        return self._smoothed_error_rad

    def _get_wrist_yaw_position(self) -> float | None:
        """Get current wrist yaw position in radians.
        
        Returns:
            Wrist yaw angle in radians, or None if unavailable.
        """
        try:
            return float(self.robot.end_of_arm.get_joint("wrist_yaw").status["pos"])
        except Exception:
            return None

    def get_wrist_yaw_position(self) -> float | None:
        """Public accessor for wrist yaw position in radians."""
        return self._get_wrist_yaw_position()
    
    def _command_velocity(self, desired_vel: float) -> None:
        """Command wrist velocity with acceleration limiting.
        
        Args:
            desired_vel: Desired velocity in rad/s.
        """
        now_s = time.time()
        dt = now_s - self._last_command_time_s
        if dt < self.min_command_interval_s:
            return
        
        dt = max(dt, 1e-3)
        desired_vel = float(
            np.clip(desired_vel, -self.max_wrist_speed_rad_s, self.max_wrist_speed_rad_s)
        )
        
        # Acceleration limit prevents sudden sign flips from noisy detections
        max_dv = self.max_wrist_accel_rad_s2 * dt
        vel_step = float(np.clip(desired_vel - self._last_cmd_vel, -max_dv, max_dv))
        cmd_vel = self._last_cmd_vel + vel_step
        
        self._send_wrist_velocity(cmd_vel)
        self._last_cmd_vel = cmd_vel
        self._last_command_time_s = now_s
    
    def update_control_from_angle_error(self, angle_error_rad: float) -> None:
        """Update wrist velocity based on angle error.
        
        Args:
            angle_error_rad: Angle error in radians (positive = rotate left).
        """
        error_rad = angle_error_rad
        current_yaw = self._get_wrist_yaw_position()
        if current_yaw is not None:
            if not self.camera_follows_wrist:
                error_rad = angle_error_rad - current_yaw
            error_rad = self.clamp_wrist_yaw_error(current_yaw, error_rad)

        filtered_error = self._smooth_error(error_rad)
        if abs(filtered_error) < self.wrist_deadband_rad:
            self._command_velocity(0.0)
            return

        desired_vel = self.wrist_direction_sign * self.control_kp * filtered_error
        self._last_pre_guard_vel = desired_vel
        if current_yaw is not None:
            desired_vel = self._apply_yaw_limit_velocity_guard(
                current_yaw,
                desired_vel,
            )
        self._last_post_guard_vel = desired_vel
        self._command_velocity(desired_vel)
        self._last_seen_time_s = time.time()
    
    def command_stop(self) -> None:
        """Stop wrist motion immediately."""
        self._last_pre_guard_vel = 0.0
        self._last_post_guard_vel = 0.0
        self._command_velocity(0.0)
    
    def shutdown(self) -> None:
        """Stop wrist and shut down robot connection."""
        try:
            self._last_pre_guard_vel = 0.0
            self._last_post_guard_vel = 0.0
            self._send_wrist_velocity(0.0)
        finally:
            self.robot.stop()

    def get_velocity_debug(self) -> tuple[float, float]:
        """Get pre-guard and post-guard velocity commands."""
        return self._last_pre_guard_vel, self._last_post_guard_vel

    def get_yaw_limits(self) -> tuple[float, float]:
        """Get current yaw limits with buffer applied."""
        min_limit = self.wrist_yaw_min_rad + self.wrist_yaw_limit_buffer_rad
        max_limit = self.wrist_yaw_max_rad - self.wrist_yaw_limit_buffer_rad
        if min_limit >= max_limit:
            min_limit = self.wrist_yaw_min_rad
            max_limit = self.wrist_yaw_max_rad
        return min_limit, max_limit
