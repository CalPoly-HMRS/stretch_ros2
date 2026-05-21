import rclpy
import time
import cv2

from math import atan2, pi
from geometry_msgs.msg import Twist
import hello_helpers.hello_misc as hm
import numpy as np
import pyrealsense2 as rs

#for tag detection
TARGET_TAG_IDS = [0,2]
MARKER_SIZE_M = 0.05
SEARCH_TIMEOUT_S = 10.0

class ItemPickup(hm.HelloNode):
    def __init__(self):
        super().__init__()

    def initialize_robot(self):
        hm.HelloNode.main(
            self, "item_pickup",
            "item_pickup", 
            wait_for_first_pointcloud=False,
        )
        #
        
        # for robot movement
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.get_logger().info("Robot node started")
	#move forward movement works YAY :D // new code 
    def stop_base(self):
        msg = Twist()
        self.cmd_vel_pub.publish(msg)


    def move_forward3ft(self):
        self.get_logger().info("Moving forward 3ft")
        distance_m = 0.9144
        speed_mps = 0.10
        move_time_s = distance_m / speed_mps

        msg = Twist()
        msg.linear.x = speed_mps
        start_time = time.time()
        while time.time() - start_time < move_time_s:
            self.cmd_vel_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.05)
        
        #self.move_to_pose({"translate_mobile_base": 0.9144}, blocking=True)
        self.stop_base()
        time.sleep(1.0)
    
    #camera looking right works
    def look_right(self):
        self.get_logger().info("Looking right toward the object")

        #have head looking down right now at about -45degrees
        self.move_to_pose({"joint_head_pan": -pi/2, 
                           "joint_head_tilt": -pi/4}, 
                           blocking=True
                          )
        
    #setting up camera detection
    def setup_camera(self):
        self.get_logger().info("Starting Realsense camera")

        #
        pipeline = rs.pipeline()        
        config = rs.config()

        #initialize camera settings
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = pipeline.start(config)

        color_stream_profile = profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()

        intr = color_stream_profile.get_intrinsics()

        camera_matrix = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0,intr.fy, intr.ppy], [0.0,0.0,1.0],],
            dtype=np.float64
        )

        dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

        return pipeline, camera_matrix, dist_coeffs
    
    def estimate_marker_pose(self, selected_corners, camera_matrix, dist_coeffs):
        half = MARKER_SIZE_M / 2.0

        object_points = np.array([[-half, half, 0.0], [half, half, 0.0],
                                  [half, -half, 0.0], [-half, -half, 0.0],],
                                  dtype=np.float32)
        
        image_points = selected_corners.reshape(4,2).astype(np.float32)

        success,rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix,
                                          dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        
        if not success:
            return None, None
        
        return rvec.reshape(3), tvec.reshape(3)
    
    def detect_aruco_tag(self):
        self.get_logger().info("searching for ArUco tag")

        pipeline = None

        try:
            pipeline, camera_matrix, dist_coeffs, = self.setup_camera()

            aruco_dict = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_6X6_250
            )

            detector_params = cv2.aruco.DetectorParameters()

            use_new_detector = hasattr(cv2.aruco, "ArucoDetector")

            if use_new_detector:
                detector = cv2.aruco.ArucoDetector(
                    aruco_dict, detector_params
                )
            else:
                detector = None
            
            start_time = time.time()

            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.01)

                if time.time() - start_time > SEARCH_TIMEOUT_S:
                    self.get_logger().error("Timed out. No ArUco tag found. ")
                    return False
                
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)

                except RuntimeError:
                    continue

                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                image = np.asanyarray(color_frame.get_data())
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                if use_new_detector:
                    corners, ids, _ = detector.detectMarkers(gray)
                else:
                    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
                
                if ids is None or len(ids) == 0:
                    continue

                flat_ids = ids.flatten().tolist()

                selected_index = None
                selected_id = None

                for wanted_id in TARGET_TAG_IDS:
                    for i, marker_id in enumerate(flat_ids):
                        if marker_id == wanted_id:
                            selected_index = i
                            selected_id = marker_id
                            break
                    
                    if selected_index is not None:
                        break
                if selected_index is None:
                    self.get_logger().info(f"Saw tag IDs {flat_ids}, but not target IDS {TARGET_TAG_IDS}")
                    continue

                selected_corners = np.array(corners[selected_index],
                                            dtype= np.float32)
                
                rvec, tvec = self.estimate_marker_pose(selected_corners, camera_matrix, dist_coeffs)

                if tvec is None:
                    continue

                # tvec[0] = left/right position of tag from camera
                # tvec[1] = up/down position of tag from camera
                # tvec[2] = forward distance from camera

                angle_error = atan2(float(tvec[0]), float(tvec[2]))

                self.get_logger().info(f"Found target tag ID: {selected_id}")
                self.get_logger().info(f"Tag position tvec: {tvec}")
                self.get_logger().info(f"Angle error: {angle_error:.3f}rad")

                return True
        finally:
            if pipeline is not None:
                pipeline.stop()




    #
    def pickup_from_right_side(self):
        self.get_logger().info("Starting right side pickup motion")

        #open the gripper
        self.move_to_pose({"joint_gripper_finger_left": 0.3},
                          blocking=False)
        time.sleep(0.5)
        self.get_logger().info("Done with gripper, starting arm movement")
        
        #turn wrist & gripper toward the right side #changed wrist_extension from .15 to .2
        #maybe change joint_lift? not sure yet 
       # self.move_to_pose({"joint_lift": 0.55,"joint_arm": 0.3, "joint_wrist_yaw": 0.0,
                     #      "joint_wrist_pitch": 0.0, "joint_wrist_roll": 0.0},
                         #  blocking=True)
        self.get_logger().info("starting arm motion")
        #joint_lift moves arm up and down
        self.move_to_pose({"joint_lift": 0.55}, blocking=False)

        #Tilt wrist downward
        self.get_logger().info("Tilting wrist down")
        self.move_to_pose({"joint_wrist_pitch": -pi/4, "joint_wrist_roll": 0.0}, 
                          blocking=False)
        


        #print statement to confirm movements
        self.get_logger().info("Wrist movements confirmed passed ")
        
        #reach outward toward object
        #currently at .3
        self.move_to_pose({"joint_arm": 0.3}, blocking=False)
        #wait
        time.sleep(0.5)
        #closing gripper
        self.move_to_pose({"joint_gripper_finger_left": 0.0}, blocking=False)
        time.sleep(0.5)

        #lift object a bit
        self.get_logger().info("Moving arm up ")
        self.move_to_pose({"joint_lift": 0.70}, blocking=False)

        #restract arm
        #self.move_to_pose({"joint_arm": 0.2}, blocking=False)

        self.get_logger().info("pickup motion complete ")
def main():
    

    node = ItemPickup()
    initialized = False

    try: 
        node.initialize_robot()
        initialized = True 

        node.move_forward3ft()

        #head lookings right
        node.look_right()

        #aruco detection
        tag_found = node.detect_aruco_tag()

        if tag_found:
            node.pickup_from_right_side()
        else:
            node.get_logger().error("No target tag detected. Skipping pickup. ")

        #move forward again after pickup attempt

        node.move_forward3ft() 
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
    
if __name__ == "__main__":
    main()