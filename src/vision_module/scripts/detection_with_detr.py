#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
from PIL import Image
import tf
import rospy
from geometry_msgs.msg import TransformStamped
import math
from pupil_apriltags import Detector
from transformer import DETRdemo, detr, transform, CLASSES
from transformer import detect as detect_from_transformer
from visualization_msgs.msg import Marker
import cv2
import tf.transformations as tf_trans
import tf2_ros
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge


def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0
    return roll, pitch, yaw

def rotation_matrix_from_euler_angles(roll, pitch, yaw):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return R_z @ R_y @ R_x

def intersection(pixel_coords, camera_intrinsics, camera_position, camera_orientation):
    fx, fy = camera_intrinsics.fx, camera_intrinsics.fy
    cx, cy = camera_intrinsics.ppx, camera_intrinsics.ppy

    u, v = pixel_coords
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = 1.0

    ray_camera = np.array([x_cam, y_cam, z_cam])
    ray_camera = ray_camera / np.linalg.norm(ray_camera)

    roll, pitch, yaw = camera_orientation
    R_world_to_camera = rotation_matrix_from_euler_angles(roll, pitch, yaw).T

    ray_world = R_world_to_camera @ ray_camera

    plane_normal = np.array([0, 0, 1])
    plane_point = np.array([0, 0, 0])

    camera_pos = np.array(camera_position)
    denom = np.dot(plane_normal, ray_world)
    if abs(denom) < 1e-6:
        raise ValueError("射線與平面平行，無法計算交點")

    t = np.dot(plane_normal, (plane_point - camera_pos)) / denom
    intersection_point = camera_pos + t * ray_world

    return intersection_point

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 獲取相機內參數
color_stream = profile.get_stream(rs.stream.color)  # 獲取 color stream 的 profile
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()  # 取得內參數
fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy
print(f"Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

# 初始化 AprilTag 檢測器
detector = Detector(families="tag36h11")

rospy.init_node('find_object_point', anonymous=False)
tf_broadcaster = tf2_ros.TransformBroadcaster()

marker_pub = rospy.Publisher('/object_marker', Marker, queue_size=10)
image_pub = rospy.Publisher('/camera/color/image_raw', RosImage, queue_size=10)
bridge = CvBridge()


while not rospy.is_shutdown():
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    

    color_image = np.asanyarray(color_frame.get_data())
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(color_image) #給detr的格式

    # Convert the RealSense image to ROS image format and publish
    ros_image = bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
    ros_image.header.stamp = rospy.Time.now()
    ros_image.header.frame_id = "camera_realtime"
    image_pub.publish(ros_image)
    
    tags = detector.detect(gray_image, estimate_tag_pose=True, camera_params=[fx, fy, cx, cy], tag_size=0.05)

    camera_position = None
    camera_orientation = None

    if tags:
        for tag in tags:
            # 提取 Pose 資訊
            pose_r = tag.pose_R  # Rotation matrix (3x3)
            pose_t = tag.pose_t  # Translation vector (3x1)

            # 構造 4x4 齊次變換矩陣
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = pose_r  # 將旋轉矩陣放入前 3x3
            transform_matrix[:3, 3] = pose_t.flatten()  # 將平移向量放入第 4 列

            # 將 4x4 矩陣轉換為四元數
            quaternion = tf_trans.quaternion_from_matrix(transform_matrix)
            
            # 建立 TF 消息
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "world"
            t.child_frame_id = "camera"
            
            t.transform.translation.x = pose_t[0][0]
            t.transform.translation.y = pose_t[1][0]
            t.transform.translation.z = pose_t[2][0]
            t.transform.rotation.x = quaternion[0]
            t.transform.rotation.y = quaternion[1]
            t.transform.rotation.z = quaternion[2]
            t.transform.rotation.w = quaternion[3]

            # 設置相機的位置與朝向
            camera_position = pose_t.flatten().tolist()
            roll, pitch, yaw = rotation_matrix_to_euler_angles(pose_r)
            camera_orientation = (roll, pitch, yaw)

            # 發布 TF
            tf_broadcaster.sendTransform(t)

            print(f"Published TF for Tag ID: {tag.tag_id}")
    else:
        rospy.logwarn("No ARTag detected in the frame.")
        continue
    
    scores, boxes = detect_from_transformer(pil_image, detr, transform)

    for score, box in zip(scores, boxes):
        cl = score.argmax()
        if cl == CLASSES.index('cup'):
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            object_world_position = intersection(
                (center_x, center_y),
                intrinsics,
                camera_position,
                camera_orientation
            )

            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "object"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = object_world_position[0]
            marker.pose.position.y = object_world_position[1]
            marker.pose.position.z = object_world_position[2]
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_pub.publish(marker)
            print(f"Object world position: {object_world_position}")


    rospy.sleep(0.1)
