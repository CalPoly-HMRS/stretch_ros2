#!/usr/bin/env python3
"""Main entry point for ArUco tracking with Stretch robot."""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

import config
from aruco_detector import (
    ArucoDetector,
    compute_selected_marker_angle,
    estimate_single_marker_pose,
    rotate_camera_matrix_90_clockwise,
)
from camera import CameraManager
from robot_controller import StretchWristTracker
from tf_publisher import TfLookupHelper, TfPublisher, TfPublisherConfig
from visualization import Visualizer


def main() -> int:
    """Main entry point for the tracking application.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Initialize camera
    profile_index = (
        config.CAMERA_PROFILE_WRIST_INDEX
        if config.DEVICE_INDEX == 1
        else config.CAMERA_PROFILE_HEAD_INDEX
    )
    camera = CameraManager(
        device_index=config.DEVICE_INDEX,
        profile_index=profile_index,
    )
    
    if not camera.initialize():
        return 1
    
    # Get camera intrinsics
    intrinsics = camera.get_intrinsics()
    if intrinsics is None:
        print("Failed to get camera intrinsics.")
        camera.stop()
        return 1
    
    camera_matrix, dist_coeffs = intrinsics
    pose_camera_matrix = camera_matrix
    
    if config.DEVICE_INDEX == 0:
        pose_camera_matrix = rotate_camera_matrix_90_clockwise(
            camera_matrix,
            camera.target_width,
            camera.target_height,
        )
    
    # Initialize ArUco detector
    detector = ArucoDetector()

    # Initialize ROS2 TF publisher (optional)
    tf_node = None
    tf_publisher = None
    tf_lookup = None
    rclpy = None
    if config.ENABLE_TF:
        try:
            import rclpy as _rclpy

            rclpy = _rclpy
            rclpy.init(args=None)
            tf_node = rclpy.create_node("aruco_tf_publisher")

            camera_frame_id = (
                config.CAMERA_FRAME_ID_WRIST
                if config.DEVICE_INDEX == 1
                else config.CAMERA_FRAME_ID_HEAD
            )
            tf_publisher = TfPublisher(
                tf_node,
                TfPublisherConfig(
                    base_frame_id=config.BASE_FRAME_ID,
                    camera_frame_id=camera_frame_id,
                    marker_frame_prefix=config.MARKER_FRAME_PREFIX,
                    publish_base_to_camera_identity=config.PUBLISH_BASE_TO_CAMERA_IDENTITY,
                ),
            )
            if config.ENABLE_TF_LOOKUP:
                tf_lookup = TfLookupHelper(tf_node)
        except Exception as exc:
            print(f"TF publishing disabled (ROS2 not available): {exc}")
            tf_node = None
            tf_publisher = None
            tf_lookup = None
    
    # Initialize robot controller
    try:
        camera_follows_wrist = config.DEVICE_INDEX == 1
        deadband = (
            config.WRIST_DEADBAND_RAD_WITH_CAMERA_FOLLOW
            if camera_follows_wrist
            else config.WRIST_DEADBAND_RAD_WITHOUT_CAMERA_FOLLOW
        )
        
        tracker = StretchWristTracker(
            command_hz=config.COMMAND_HZ,
            camera_follows_wrist=camera_follows_wrist,
            wrist_direction_sign=config.WRIST_DIRECTION_SIGN,
            head_pan_to_wrist_yaw_offset_rad=config.HEAD_PAN_TO_WRIST_YAW_OFFSET_RAD,
            deadband_rad=deadband,
            wrist_yaw_min_rad=config.WRIST_YAW_MIN_RAD,
            wrist_yaw_max_rad=config.WRIST_YAW_MAX_RAD,
            wrist_yaw_limit_buffer_rad=config.WRIST_YAW_LIMIT_BUFFER_RAD,
            control_kp=config.CONTROL_KP,
            max_wrist_speed_rad_s=config.MAX_WRIST_SPEED_RAD_S,
            max_wrist_accel_rad_s2=config.MAX_WRIST_ACCEL_RAD_S2,
            error_smoothing_alpha=config.ERROR_SMOOTHING_ALPHA,
        )
        tracker.initialize_robot()
    except Exception as exc:
        print(f"Failed to initialize Stretch control: {exc}")
        camera.stop()
        return 1
    
    # Initialize display
    visualizer = Visualizer(window_name=config.WINDOW_NAME)
    
    print("Full live tracking started. Press 'q' in the video window to quit.")
    print(f"Camera: {camera.target_width}x{camera.target_height}@{camera.target_fps}fps")
    print(f"Tracking IDs: {list(config.TARGET_TAG_IDS) if config.TARGET_TAG_IDS else 'ANY'}")
    print(f"Camera follows wrist: {config.DEVICE_INDEX == 1}")
    
    # Main tracking loop
    frame_count = 0
    fps_timer_start = time.time()
    measured_fps = 0.0
    last_no_frame_log_s = 0.0
    seen_first_frame = False
    
    try:
        while True:
            # Capture frame
            image = camera.get_frame(timeout_ms=config.CAMERA_TIMEOUT_MS)
            if image is None:
                now_s = time.time()
                if (now_s - last_no_frame_log_s) > config.NO_FRAME_LOG_INTERVAL_S:
                    print("Waiting for camera frames...")
                    last_no_frame_log_s = now_s
                continue
            
            if not seen_first_frame:
                print("Received first camera frame. Display loop is running.")
                seen_first_frame = True
            
            # Rotate image if needed (head camera)
            if config.DEVICE_INDEX == 0:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
            # Detect markers
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners, ids = detector.detect_markers(gray_image)
            marker_count = 0 if ids is None else len(ids)
            
            # Draw detected markers
            visualizer.draw_detected_markers(image, corners, ids)

            # Publish TF frames (if enabled)
            if tf_publisher is not None and tf_node is not None:
                if rclpy is not None and tf_lookup is not None:
                    rclpy.spin_once(tf_node, timeout_sec=0.0)
                stamp = tf_node.get_clock().now().to_msg()
                tf_publisher.publish_base_to_camera_identity(stamp)
            
            # Compute tracking
            selected_id = None
            angle_error = None
            rvec = None
            tvec = None
            
            if marker_count > 0:
                angle_error, selected_id, rvec, tvec = compute_selected_marker_angle(
                    corners,
                    ids,
                    pose_camera_matrix,
                    dist_coeffs,
                    config.MARKER_SIZE_M,
                    tuple(config.TARGET_TAG_IDS),
                )

                if tf_publisher is not None and tf_node is not None:
                    if config.PUBLISH_ALL_MARKERS and ids is not None:
                        flat_ids = ids.flatten().tolist()
                        for i, marker_id in enumerate(flat_ids):
                            marker_corners = np.array(corners[i], dtype=np.float32)
                            marker_rvec, marker_tvec = estimate_single_marker_pose(
                                marker_corners,
                                config.MARKER_SIZE_M,
                                pose_camera_matrix,
                                dist_coeffs,
                            )
                            if marker_rvec is not None and marker_tvec is not None:
                                tf_publisher.publish_marker(
                                    marker_id,
                                    marker_rvec,
                                    marker_tvec,
                                    stamp,
                                )
                    elif rvec is not None and tvec is not None and selected_id is not None:
                        tf_publisher.publish_marker(selected_id, rvec, tvec, stamp)
                
                used_tf_control = False
                if (
                    config.USE_TF_FOR_CONTROL
                    and tf_lookup is not None
                    and selected_id is not None
                ):
                    marker_frame = f"{config.MARKER_FRAME_PREFIX}{selected_id}"
                    transform = tf_lookup.lookup_transform(
                        config.BASE_FRAME_ID,
                        marker_frame,
                        timeout_s=config.TF_LOOKUP_TIMEOUT_S,
                    )
                    if transform is not None:
                        translation = transform.transform.translation
                        tracker.update_control_from_target_translation(
                            translation.x,
                            translation.z,
                        )
                        used_tf_control = True

                if angle_error is not None and not used_tf_control:
                    # Draw marker pose
                    if rvec is not None and tvec is not None:
                        visualizer.draw_marker_axes(
                            image,
                            pose_camera_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                            config.AXIS_LENGTH,
                        )
                    
                    # Draw angle error indicator
                    visualizer.draw_angle_indicator(
                        image,
                        angle_error,
                        config.ERROR_VISUALIZATION_LENGTH,
                    )
                    
                    # Update robot control
                    tracker.update_control_from_angle_error(angle_error)
                else:
                    if not used_tf_control:
                        tracker.command_stop()
            else:
                tracker.command_stop()
            
            # Update FPS measurement
            frame_count += 1
            elapsed = time.time() - fps_timer_start
            if elapsed >= config.FPS_MEASUREMENT_INTERVAL_S:
                measured_fps = frame_count / elapsed
                frame_count = 0
                fps_timer_start = time.time()
            
            # Draw HUD
            status = "TRACKING" if angle_error is not None else "SEARCHING"
            wrist_yaw = None
            if config.SHOW_WRIST_YAW:
                wrist_yaw = tracker.get_wrist_yaw_position()
            pre_guard_vel = None
            post_guard_vel = None
            if config.SHOW_VELOCITY_DEBUG:
                pre_guard_vel, post_guard_vel = tracker.get_velocity_debug()
            yaw_limits = None
            if config.SHOW_YAW_LIMITS:
                yaw_limits = tracker.get_yaw_limits()
            visualizer.draw_hud(
                image,
                measured_fps,
                marker_count,
                status,
                selected_id=selected_id,
                angle_error=angle_error,
                wrist_yaw=wrist_yaw,
                tvec=tvec,
                pre_guard_vel=pre_guard_vel,
                post_guard_vel=post_guard_vel,
                yaw_limits=yaw_limits,
                show_fps=config.SHOW_FPS,
                show_marker_count=config.SHOW_MARKER_COUNT,
                show_status=config.SHOW_STATUS,
                show_selected_id=config.SHOW_SELECTED_ID,
                show_angle_error=config.SHOW_ANGLE_ERROR,
                show_wrist_yaw=config.SHOW_WRIST_YAW,
                show_tvec=config.SHOW_TVEC,
                show_velocity_debug=config.SHOW_VELOCITY_DEBUG,
                show_yaw_limits=config.SHOW_YAW_LIMITS,
            )
            
            # Display frame
            if not visualizer.show_frame(image):
                break
    
    finally:
        try:
            tracker.shutdown()
        except Exception:
            pass
        if rclpy is not None and tf_node is not None:
            try:
                tf_node.destroy_node()
            finally:
                rclpy.shutdown()
        camera.stop()
        visualizer.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
