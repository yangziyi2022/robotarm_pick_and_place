#!/usr/bin/env python3
import pyrealsense2 as rs
from pupil_apriltags import Detector
import numpy as np
import cv2
import tf.transformations as tf_trans
import rospy
from geometry_msgs.msg import TransformStamped
import tf2_ros

# 初始化 RealSense
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

# 初始化 ROS 節點與 TF 發布器
rospy.init_node('artag_tf_broadcaster')
tf_broadcaster = tf2_ros.TransformBroadcaster()

while not rospy.is_shutdown():
    # 獲取影像幀
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    # 將影像轉為灰度
    color_image = np.asanyarray(color_frame.get_data())
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # 檢測 ARTag
    tags = detector.detect(gray_image, estimate_tag_pose=True, camera_params=[fx, fy, cx, cy], tag_size=0.05)

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

            # 發布 TF
            tf_broadcaster.sendTransform(t)

            print(f"Published TF for Tag ID: {tag.tag_id}")

