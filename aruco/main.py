"""Run the standalone ArUco detection loop."""

from __future__ import annotations

import time

import cv2
import numpy as np
import rclpy

from aruco__tf_publisher import ArucoTfPublisher
from aruco_detector import ArucoDetector, estimate_single_marker_pose, rotate_camera_matrix_90_clockwise
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


def _select_marker_index(ids: list[int], target_ids: tuple[int, ...]) -> int | None:
	if not ids:
		return None
	if target_ids:
		for wanted_id in target_ids:
			for i, marker_id in enumerate(ids):
				if marker_id == wanted_id:
					return i
		return None
	return 0


def _rotate_corners_90_clockwise(
	corners: list,
	image_width: int,
	image_height: int,
) -> list:
	rotated = []
	for corner in corners:
		pts = corner.reshape(-1, 2)
		rotated_pts = np.empty_like(pts)
		rotated_pts[:, 0] = (image_height - 1) - pts[:, 1]
		rotated_pts[:, 1] = pts[:, 0]
		rotated.append(rotated_pts.reshape(corner.shape))
	return rotated


def main() -> None:
	rclpy.init()
	node = rclpy.create_node("aruco_detector")
	device_index = config.DEVICE_INDEX
	if config.PROMPT_FOR_CAMERA_SELECTION:
		device_index = _prompt_for_device_index(config.DEVICE_INDEX)
	is_head_camera = device_index == 0
	parent_frame = "camera_link" if is_head_camera else "gripper_camera_link"
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

			flat_ids = [] if ids is None else ids.flatten().tolist()
			selected_idx = _select_marker_index(flat_ids, target_ids)
			selected_id = None if selected_idx is None else flat_ids[selected_idx]

			rvec = None
			tvec = None
			if selected_idx is not None:
				selected_corners = corners[selected_idx]
				rvec, tvec = estimate_single_marker_pose(
					selected_corners,
					config.MARKER_SIZE_M,
					camera_matrix,
					dist_coeffs,
				)

			marker_count = 0 if ids is None else len(ids)
			status = "TRACKING" if marker_count > 0 else "SEARCHING"

			display_frame = frame
			display_corners = corners
			display_camera_matrix = camera_matrix
			if is_head_camera:
				display_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
				display_camera_matrix = rotate_camera_matrix_90_clockwise(
					camera_matrix,
					display_frame.shape[1],
					display_frame.shape[0],
				)
				if corners is not None:
					display_corners = _rotate_corners_90_clockwise(
						corners,
						frame.shape[1],
						frame.shape[0],
					)

			visualizer.draw_detected_markers(display_frame, display_corners, ids)

			if rvec is not None and tvec is not None:
				visualizer.draw_marker_axes(
					display_frame,
					display_camera_matrix,
					dist_coeffs,
					rvec,
					tvec,
					axis_length=config.AXIS_LENGTH,
				)
				if selected_id is not None:
					tf_publisher.publish(rvec, tvec, selected_id)

			visualizer.draw_hud(
				display_frame,
				fps=fps,
				marker_count=marker_count,
				status=status,
				selected_id=selected_id,
				angle_error=None,
				wrist_yaw=None,
				tvec=tvec,
				pre_guard_vel=None,
				post_guard_vel=None,
				yaw_limits=None,
				show_fps=config.SHOW_FPS,
				show_marker_count=config.SHOW_MARKER_COUNT,
				show_status=config.SHOW_STATUS,
				show_selected_id=config.SHOW_SELECTED_ID,
				show_angle_error=False,
				show_wrist_yaw=False,
				show_tvec=config.SHOW_TVEC,
				show_velocity_debug=False,
				show_yaw_limits=False,
			)

			if not visualizer.show_frame(display_frame):
				break
	finally:
		camera.stop()
		visualizer.cleanup()
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
