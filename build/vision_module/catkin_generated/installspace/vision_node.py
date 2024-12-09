#!/usr/bin/env python3
import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
from geometry_msgs.msg import Point

def vision_node():
    rospy.init_node('vision_node', anonymous=True)
    pub = rospy.Publisher('/robot_target', Point, queue_size=10)
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
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv_image, lower_white, upper_white)

            # 形态学操作
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


            # 尋找輪廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    depth = depth_frame.get_distance(center_x, center_y)
                    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [center_x, center_y], depth)
                    print(f"目標點: {point_3d}")

                    # 發布目標點到 /robot_target
                    msg = Point()
                    msg.x, msg.y, msg.z = point_3d
                    pub.publish(msg)

                    # 在圖像中顯示目標物
                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)


            cv2.imshow("Color Image", color_image)
            cv2.imshow("Mask", mask)
            if cv2.waitKey(1) == 27:
                break

    finally:
        pipeline.stop()

if __name__ == '__main__':
    vision_node()
