#!/usr/bin/env python3
import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
import tf2_ros
import geometry_msgs.msg

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point

def publish_tf_transform(parent_frame, child_frame, translation):
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = parent_frame # 父坐標系，如 "camera_frame"
    t.child_frame_id = child_frame # 子坐標系，如 "object_frame"

    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]

    # 假設無旋轉，設置為單位四元數
    t.transform.rotation.x = 0.0
    t.transform.rotation.y = 0.0
    t.transform.rotation.z = 0.0
    t.transform.rotation.w = 1.0

    br.sendTransform(t)

from visualization_msgs.msg import Marker

def publish_marker(position):
    marker_pub = rospy.Publisher('/object_marker', Marker, queue_size=10)
    marker = Marker()
    marker.header.frame_id = "camera_frame"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "object_marker"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    # 設置物體位置
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # 設置 Marker 尺寸
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05

    # 設置 Marker 顏色
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    marker_pub.publish(marker)

    


def vision_node():
    rospy.init_node('vision_node', anonymous=True)
    pub = rospy.Publisher('/robot_target', Point, queue_size=10)
    pointcloud_pub = rospy.Publisher('/vision_pointcloud', PointCloud2, queue_size=10)
    rate = rospy.Rate(10)

    # 初始化 RealSense 相機
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    try:
        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # RGB 圖像
            color_image = np.asanyarray(color_frame.get_data())
            blurred = cv2.GaussianBlur(color_image, (5, 5), 0)

            # 顏色檢測（白色物體）
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            lower_white = np.array([20, 100, 67])
            upper_white = np.array([30, 255, 255])
            mask = cv2.inRange(hsv_image, lower_white, upper_white)

            # 形态学操作
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


            # 尋找輪廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 蒐集點
            points = []

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    # depth = depth_frame.get_distance(center_x, center_y)
                    depth_values = [
                        depth_frame.get_distance(center_x + dx, center_y + dy)
                        for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
                    ]
                     # 過濾合理範圍內的深度值
                    valid_depths = [d for d in depth_values if 0.2 < d < 10]

                    if valid_depths:
                        depth = np.mean(valid_depths)
                        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [center_x, center_y], depth)
                        points.append([point_3d[0], point_3d[1], point_3d[2]])
                        print(f"目標點: {point_3d}")
                    else:
                        print("深度超出範圍")

                    # 發布目標點到 /robot_target
                    msg = Point()
                    msg.x, msg.y, msg.z = point_3d
                    pub.publish(msg)

                    # # 發布 Marker 到 /object_marker
                    # publish_marker(point_3d)

                    # 在圖像中顯示目標物
                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)

            # 創建PointCloud2 消息
            header = rospy.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera_frame"
            cloud_msg = point_cloud2.create_cloud_xyz32(header, points)
            pointcloud_pub.publish(cloud_msg)
            

            cv2.imshow("Color Image", color_image)
            cv2.imshow("Mask", mask)
            if cv2.waitKey(1) == 27:
                break

    finally:
        pipeline.stop()

if __name__ == '__main__':
    vision_node()
