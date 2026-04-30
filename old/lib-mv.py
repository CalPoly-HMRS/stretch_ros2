#!/usr/bin/env python3

from __future__ import annotations

import sys
import time
from math import atan2, pi

import cv2
import numpy as np
import pyrealsense2 as rs

try:
    import stretch_body.robot as stretch_robot
except ImportError:
    stretch_robot = None


# ArUco IDs to track, in priority order.
# Example: [23, 42, 7] means track ID 23 first, then 42, then 7 if visible.
# Set to [] to track the first detected marker of any ID.
TARGET_TAG_IDS: list[int] = [0, 2]

# Set to -1.0 if wrist turns the wrong way for positive angle error.
WRIST_DIRECTION_SIGN: float = -1.0


class StretchWristTracker:
    def __init__(
        self,
        command_hz: float = 20.0,
        camera_follows_wrist: bool = False,
    ) -> None:
        if stretch_robot is None:
            raise RuntimeError(
                "stretch_body is not installed. Install it on the robot before running this script."
            )

        self.robot = stretch_robot.Robot()
        self.camera_follows_wrist = camera_follows_wrist

        self.wrist_deadband_rad = 0.01 if camera_follows_wrist else 0.012
        self.wrist_direction_sign = WRIST_DIRECTION_SIGN

        # Velocity-loop gains and limits for smooth but stable tracking.
        self.control_kp = 2.0
        self.max_wrist_speed_rad_s = 1.8
        self.max_wrist_accel_rad_s2 = 6.0
        self.error_smoothing_alpha = 0.25

        self.min_command_interval_s = 1.0 / command_hz
        self._last_command_time_s = 0.0
        self._smoothed_error_rad = 0.0
        self._last_cmd_vel = 0.0
        self._last_seen_time_s = 0.0

    @staticmethod
    def wrap_angle(angle: float) -> float:
        while angle > pi:
            angle -= 2.0 * pi
        while angle < -pi:
            angle += 2.0 * pi
        return angle

    def initialize_robot(self) -> None:
        did_startup = self.robot.startup()
        if not did_startup:
            raise RuntimeError("Failed to connect to Stretch hardware via stretch_body.robot.Robot().")

        print("Connected to Stretch hardware.")
        if not self.robot.is_homed():
            print("Warning: Robot is not homed. Tracking may be inaccurate until homed.")

        # Move to neutral tracking pose.
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
        try:
            self.robot.end_of_arm.get_joint("wrist_yaw").set_velocity(float(cmd_vel))
        except Exception:
            return

    def _smooth_error(self, raw_error_rad: float) -> float:
        alpha = self.error_smoothing_alpha
        self._smoothed_error_rad = ((1.0 - alpha) * self._smoothed_error_rad) + (alpha * raw_error_rad)
        return self._smoothed_error_rad

    def _command_velocity(self, desired_vel: float) -> None:
        now_s = time.time()
        dt = now_s - self._last_command_time_s
        if dt < self.min_command_interval_s:
            return

        dt = max(dt, 1e-3)
        desired_vel = float(np.clip(desired_vel, -self.max_wrist_speed_rad_s, self.max_wrist_speed_rad_s))

        # Acceleration limit prevents sudden sign flips from noisy detections.
        max_dv = self.max_wrist_accel_rad_s2 * dt
        vel_step = float(np.clip(desired_vel - self._last_cmd_vel, -max_dv, max_dv))
        cmd_vel = self._last_cmd_vel + vel_step

        self._send_wrist_velocity(cmd_vel)
        self._last_cmd_vel = cmd_vel
        self._last_command_time_s = now_s

    def update_control_from_angle_error(self, angle_error_rad: float) -> None:
        filtered_error = self._smooth_error(angle_error_rad)

        if abs(filtered_error) < self.wrist_deadband_rad:
            self._command_velocity(0.0)
            return

        desired_vel = self.wrist_direction_sign * self.control_kp * filtered_error
        self._command_velocity(desired_vel)
        self._last_seen_time_s = time.time()

    def command_stop(self) -> None:
        self._command_velocity(0.0)

    def shutdown(self) -> None:
        try:
            self._send_wrist_velocity(0.0)
        finally:
            self.robot.stop()


def get_intrinsics_matrix_and_dist(color_profile: rs.video_stream_profile) -> tuple[np.ndarray, np.ndarray]:
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
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    _ = image_width
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
    if ids is None or len(ids) == 0:
        return None, None, None, None

    flat_ids = ids.flatten().tolist()

    selected_idx = None
    selected_id = None
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

    angle_error = atan2(float(tvec[0]), float(tvec[2]))
    return angle_error, selected_id, rvec, tvec


def main() -> int:
    ctx = rs.context()
    devices = list(ctx.query_devices())

    if not devices:
        print("No RealSense devices detected.")
        return 1

    # 0 for head camera, 1 for wrist camera
    device_idx = 1
    device = devices[device_idx]

    selected_serial = device.get_info(rs.camera_info.serial_number)
    camera_follows_wrist = device_idx == 1

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

    profiles = sorted(profiles, key=lambda x: (x[0] * x[1], x[2], str(x[3])))
    if not profiles:
        print("No color stream profiles found for the selected device.")
        return 1

    if device_idx == 1:
        print("Using wrist camera.")
        profile = profiles[18]  # 424x240 @ 60 fps | bgr8 for this setup
    else:
        print("Using head camera.")
        profile = profiles[150]  # 960x540 @ 60 fps | bgr8 for this setup

    print(f"Selected profile: {profile[0]}x{profile[1]} @ {profile[2]} fps | {profile[3]}")

    width, height, fps, fmt = profile
    target_tag_ids = tuple(TARGET_TAG_IDS)
    marker_size_m = 0.05

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected_serial)
    config.enable_stream(rs.stream.color, width, height, fmt, fps)

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        print(f"Failed to start stream at {width}x{height}@{fps}, format={fmt}: {exc}")
        return 1

    color_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    camera_matrix, dist_coeffs = get_intrinsics_matrix_and_dist(color_stream_profile)

    pose_camera_matrix = camera_matrix
    if device_idx == 0:
        pose_camera_matrix = rotate_camera_matrix_90_clockwise(camera_matrix, width, height)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    try:
        tracker = StretchWristTracker(camera_follows_wrist=camera_follows_wrist)
        tracker.initialize_robot()
    except Exception as exc:
        print(f"Failed to initialize Stretch control: {exc}")
        pipeline.stop()
        return 1

    print("Full live tracking started. Press 'q' in the video window to quit.")
    print(f"Selected mode: {width}x{height}@{fps}, format={fmt}")
    print(f"Tracking IDs: {list(target_tag_ids) if target_tag_ids else 'ANY'}")
    print(f"Camera follows wrist: {camera_follows_wrist}")

    window_name = "Full Live Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    fps_timer_start = time.time()
    measured_fps = 0.0
    last_no_frame_log_s = 0.0
    seen_first_frame = False

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1500)
            except RuntimeError:
                now_s = time.time()
                if (now_s - last_no_frame_log_s) > 2.0:
                    print("Waiting for camera frames...")
                    last_no_frame_log_s = now_s
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                now_s = time.time()
                if (now_s - last_no_frame_log_s) > 2.0:
                    print("Color frame not available yet...")
                    last_no_frame_log_s = now_s
                continue

            if not seen_first_frame:
                print("Received first camera frame. Display loop is running.")
                seen_first_frame = True

            image = np.asanyarray(color_frame.get_data())

            if device_idx == 0:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            detect_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = detector.detectMarkers(detect_image)
            marker_count = 0 if ids is None else len(ids)

            selected_id = None
            angle_error = None
            tvec_text = "tvec (x,y,z): n/a"
            if marker_count > 0:
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
                angle_error, selected_id, rvec, tvec = compute_selected_marker_angle(
                    corners,
                    ids,
                    pose_camera_matrix,
                    dist_coeffs,
                    marker_size_m,
                    target_tag_ids,
                )
                if tvec is not None:
                    tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
                    tvec_text = f"tvec (x,y,z): {tx:+.3f}, {ty:+.3f}, {tz:+.3f}"
                if angle_error is not None:
                    tracker.update_control_from_angle_error(angle_error)

                    if rvec is not None and tvec is not None:
                        cv2.drawFrameAxes(
                            image,
                            pose_camera_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                            marker_size_m * 0.5,
                        )

                    h, w = image.shape[:2]
                    start = (w // 2, h - 1)

                    error_length = 100
                    dx = int(error_length * np.sin(angle_error))
                    dy = int(error_length * np.cos(angle_error))

                    end_x = int(np.clip(start[0] + dx, 0, w - 1))
                    end_y = int(np.clip(start[1] - dy, 0, h - 1))
                    cv2.line(image, start, (end_x, end_y), (255, 0, 0), 4)
                else:
                    tracker.command_stop()
            else:
                tracker.command_stop()

            frame_count += 1
            elapsed = time.time() - fps_timer_start
            if elapsed >= 1.0:
                measured_fps = frame_count / elapsed
                frame_count = 0
                fps_timer_start = time.time()

            status_text = "TRACKING" if angle_error is not None else "SEARCHING"

            cv2.putText(image, f"Measured FPS: {measured_fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Markers: {marker_count}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Status: {status_text}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            if selected_id is not None:
                cv2.putText(image, f"Selected ID: {selected_id}", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)
            if angle_error is not None:
                cv2.putText(image, f"Angle error (rad): {angle_error:+.3f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

            cv2.putText(image, tvec_text, (10, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        try:
            tracker.shutdown()
        except Exception:
            pass
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
