"""Run the standalone ArUco detection loop."""

from __future__ import annotations

import time

import cv2
import rclpy

from aruco__tf_publisher import ArucoTfPublisher
from aruco_detector import ArucoDetector, compute_selected_marker_angle
from camera import CameraManager
import config
from visualization import Visualizer


def _pick_profile_index(device_index: int) -> int:
	if device_index == 0:
		return config.CAMERA_PROFILE_HEAD_INDEX
	return config.CAMERA_PROFILE_WRIST_INDEX


def _prompt_for_device_index(default_index: int) -> int:
	choice = input("Select camera: [0] head, [1] wrist (default: {default}): ".format(
		default=default_index
	)).strip()
	if choice == "0":
		return 0
	if choice == "1":
		return 1
	if choice:
		print("Invalid selection. Using default camera index.")
	return default_index


def main() -> None:
	rclpy.init()
	node = rclpy.create_node("aruco_detector")
	device_index = config.DEVICE_INDEX
	if config.PROMPT_FOR_CAMERA_SELECTION:
		device_index = _prompt_for_device_index(config.DEVICE_INDEX)
	parent_frame = "camera_link" if device_index == 0 else "gripper_camera_link"
	tf_publisher = ArucoTfPublisher(node, parent_frame=parent_frame)

	camera = CameraManager(
		device_index=device_index,
		profile_index=_pick_profile_index(device_index),
	)
	if not camera.initialize():
		node.destroy_node()
		rclpy.shutdown()
		return

	intrinsics = camera.get_intrinsics()
	if intrinsics is None:
		camera.stop()
		node.destroy_node()
		rclpy.shutdown()
		return
	camera_matrix, dist_coeffs = intrinsics

	detector = ArucoDetector()
	visualizer = Visualizer(window_name=config.WINDOW_NAME)

	target_ids = tuple(config.TARGET_TAG_IDS)
	last_fps_time = time.time()
	frame_count = 0
	fps = 0.0
	last_frame_time = time.time()

	try:
		while True:
			frame = camera.get_frame(timeout_ms=config.CAMERA_TIMEOUT_MS)
			now = time.time()

			if frame is None:
				if now - last_frame_time >= config.NO_FRAME_LOG_INTERVAL_S:
					print("No frame received from camera.")
					last_frame_time = now
				continue

			last_frame_time = now
			frame_count += 1

			if now - last_fps_time >= config.FPS_MEASUREMENT_INTERVAL_S:
				fps = frame_count / (now - last_fps_time)
				frame_count = 0
				last_fps_time = now

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			corners, ids = detector.detect_markers(gray)

			angle_error, selected_id, rvec, tvec = compute_selected_marker_angle(
				corners or [],
				ids,
				camera_matrix,
				dist_coeffs,
				config.MARKER_SIZE_M,
				target_ids,
			)

			marker_count = 0 if ids is None else len(ids)
			status = "TRACKING" if marker_count > 0 else "SEARCHING"

			visualizer.draw_detected_markers(frame, corners, ids)

			if rvec is not None and tvec is not None:
				visualizer.draw_marker_axes(
					frame,
					camera_matrix,
					dist_coeffs,
					rvec,
					tvec,
					axis_length=config.AXIS_LENGTH,
				)
				if selected_id is not None:
					tf_publisher.publish(rvec, tvec, selected_id)

			if angle_error is not None:
				visualizer.draw_angle_indicator(
					frame,
					angle_error,
					error_visualization_length=config.ERROR_VISUALIZATION_LENGTH,
				)

			visualizer.draw_hud(
				frame,
				fps=fps,
				marker_count=marker_count,
				status=status,
				selected_id=selected_id,
				angle_error=angle_error,
				wrist_yaw=None,
				tvec=tvec,
				pre_guard_vel=None,
				post_guard_vel=None,
				yaw_limits=None,
				show_fps=config.SHOW_FPS,
				show_marker_count=config.SHOW_MARKER_COUNT,
				show_status=config.SHOW_STATUS,
				show_selected_id=config.SHOW_SELECTED_ID,
				show_angle_error=config.SHOW_ANGLE_ERROR,
				show_wrist_yaw=False,
				show_tvec=config.SHOW_TVEC,
				show_velocity_debug=False,
				show_yaw_limits=False,
			)

			if not visualizer.show_frame(frame):
				break
	finally:
		camera.stop()
		visualizer.cleanup()
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
