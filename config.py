"""Configuration and constants for ArUco tracking."""

# https://docs.hello-robot.com/latest/python/moving/

from __future__ import annotations

from math import pi

# ArUco IDs to track, in priority order.
# Example: [23, 42, 7] means track ID 23 first, then 42, then 7 if visible.
# Set to [] to track the first detected marker of any ID.
TARGET_TAG_IDS: list[int] = [0, 2]

# Set to -1.0 if wrist turns the wrong way for positive angle error.
WRIST_DIRECTION_SIGN: float = -1.0

# Marker size in meters (50mm = 0.05m)
MARKER_SIZE_M: float = 0.05

# RealSense camera configuration
# Device index: 0 for head camera, 1 for wrist camera
DEVICE_INDEX: int = 1

# Camera profile selection
# Format: (width, height, fps, format)
# Index into sorted profiles list for each device
# Wrist camera examples
#   18  -> 424x240 @ 60 fps | bgr8
#   54  -> 640x480 @ 30 fps | bgr8
#   84  -> 1280x720 @ 15 fps | bgr8
# Head camera examples
#   150 -> 960x540 @ 60 fps | bgr8
#   186 -> 1920x1080 @ 30 fps | bgr8
CAMERA_PROFILE_HEAD_INDEX: int = 150
CAMERA_PROFILE_WRIST_INDEX: int = 18

# Wrist tracking parameters
COMMAND_HZ: float = 20.0
WRIST_DEADBAND_RAD_WITH_CAMERA_FOLLOW: float = 0.01
WRIST_DEADBAND_RAD_WITHOUT_CAMERA_FOLLOW: float = 0.012

# Joint limits (approximate; tune to your robot)
# WRIST_YAW_MIN_RAD: float = -1.39
# WRIST_YAW_MAX_RAD: float = 4.42
# WRIST_PITCH_MIN_RAD: float = -1.57
# WRIST_PITCH_MAX_RAD: float = 0.57
# WRIST_ROLL_MIN_RAD: float = -3.14
# WRIST_ROLL_MAX_RAD: float = 3.14

# EXTRA SAFETY MARGIN: Reduce max angles by this amount to avoid hitting hard limits
SAFETY_MARGIN_RAD: float = 0.1
WRIST_YAW_MIN_RAD: float = -0.75 + SAFETY_MARGIN_RAD
WRIST_YAW_MAX_RAD: float = 3.2 - SAFETY_MARGIN_RAD
WRIST_PITCH_MIN_RAD: float = -1.57 + SAFETY_MARGIN_RAD
WRIST_PITCH_MAX_RAD: float = 0.57 - SAFETY_MARGIN_RAD
WRIST_ROLL_MIN_RAD: float = -3.14 + SAFETY_MARGIN_RAD
WRIST_ROLL_MAX_RAD: float = 3.14 - SAFETY_MARGIN_RAD
ARM_MIN_M: float = 0.0
ARM_MAX_M: float = 0.52
LIFT_MIN_M: float = 0.0
LIFT_MAX_M: float = 1.1

# Velocity control gains
CONTROL_KP: float = 4.0
MAX_WRIST_SPEED_RAD_S: float = 1.8
MAX_WRIST_ACCEL_RAD_S2: float = 4.0
ERROR_SMOOTHING_ALPHA: float = 0.25

# Robot initialization pose
INITIAL_HEAD_PAN: float = -pi / 2
INITIAL_HEAD_TILT: float = -pi / 8
INITIAL_WRIST_YAW: float = 0.0
INITIAL_WRIST_PITCH: float = 0.0
INITIAL_WRIST_ROLL: float = 0.0

# Display settings
WINDOW_NAME: str = "Full Live Tracking"
FPS_MEASUREMENT_INTERVAL_S: float = 1.0
NO_FRAME_LOG_INTERVAL_S: float = 2.0
CAMERA_TIMEOUT_MS: int = 1500

# Visualization parameters
ERROR_VISUALIZATION_LENGTH: int = 100
AXIS_LENGTH: float = 0.025  # Half of marker size
