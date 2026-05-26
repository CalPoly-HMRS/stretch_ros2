"""Configuration and constants for ArUco detection."""

from __future__ import annotations

# ArUco IDs to track, in priority order.
# Example: [23, 42, 7] means track ID 23 first, then 42, then 7 if visible.
# Set to [] to track the first detected marker of any ID.
TARGET_TAG_IDS: list[int] = [0, 2]

# Marker size in meters (50mm = 0.05m)
MARKER_SIZE_M: float = 0.05


# RealSense camera configuration
# Device index: 0 for head camera, 1 for wrist camera
DEVICE_INDEX: int = 1

# Prompt for camera selection at startup.
PROMPT_FOR_CAMERA_SELECTION: bool = False

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

# Depth profile selection (same indexing scheme as color profiles)
# Set to -1 to auto-select a depth profile that best matches the color stream.
DEPTH_PROFILE_HEAD_INDEX: int = -1
DEPTH_PROFILE_WRIST_INDEX: int = -1


# Depth pose refinement mode:
# - "off": RGB-only pose from solvePnP
# - "simple": scale translation with median depth inside marker polygon
# - "plane": fit a plane to depth points and re-estimate pose on that plane
DEPTH_POSE_MODE: str = "simple"

# Depth processing settings (meters)
# Currently based on optimal performance of the Intel Realsense D435i
DEPTH_MIN_METERS: float = 0.3
DEPTH_MAX_METERS: float = 3.0
DEPTH_MIN_POINTS: int = 16

if DEVICE_INDEX == 0:  # Head camera
    DEPTH_MIN_METERS: float = 0.3
    DEPTH_MAX_METERS: float = 3.0
elif DEVICE_INDEX == 1:  # Wrist camera
    DEPTH_MIN_METERS: float = 0.07
    DEPTH_MAX_METERS: float = 0.5

# note for head camera: still 2% error at 2 meters, and recommended ideal depth resolution is 480p for some reason?
# ¯\_(ツ)_/¯
# Also the umm wrist camera (Intel RealSense D405) has a very short optimal range (sub millimeter accuracy tho!) from like 7cm to 50cm...

# Align depth to the color stream so each RGB pixel can be sampled in the depth image.
ALIGN_DEPTH_TO_COLOR: bool = True


# Display settings
SHOW_WINDOW: bool = True
WINDOW_NAME: str = "Full Live Tracking"
FPS_MEASUREMENT_INTERVAL_S: float = 1.0
NO_FRAME_LOG_INTERVAL_S: float = 2.0
CAMERA_TIMEOUT_MS: int = 1500

# Visualization parameters
ERROR_VISUALIZATION_LENGTH: int = 100
AXIS_LENGTH: float = 0.025  # Half of marker size

# HUD options
SHOW_FPS: bool = True
SHOW_MARKER_COUNT: bool = False
SHOW_STATUS: bool = True
SHOW_SELECTED_ID: bool = True
SHOW_ANGLE_ERROR: bool = True
SHOW_TVEC: bool = False
SHOW_WRIST_YAW: bool = True
SHOW_VELOCITY_DEBUG: bool = False
SHOW_YAW_LIMITS: bool = False