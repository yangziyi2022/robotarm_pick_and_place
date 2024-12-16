#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import rospy
import tf2_ros
import geometry_msgs.msg

def get_camera_intrinsics():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        camera_matrix = [
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1],
        ]
        dist_coeffs = intrinsics.coeffs
        return camera_matrix, dist_coeffs
    finally:
        pipeline.stop()

def publish_to_tf(rvec, tvec):
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "ar_tag"
    t.child_frame_id = "camera_frame"
    t.transform.translation.x = tvec[0][0]
    t.transform.translation.y = tvec[0][1]
    t.transform.translation.z = tvec[0][2]
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    quaternion = tf.transformations.quaternion_from_matrix(np.vstack((rotation_matrix, [0, 0, 0, 1])))
    t.transform.rotation.x = quaternion[0]
    t.transform.rotation.y = quaternion[1]
    t.transform.rotation.z = quaternion[2]
    t.transform.rotation.w = quaternion[3]
    br.sendTransform(t)

rospy.init_node("aruco_tf_publisher", anonymous=True)

camera_matrix, dist_coeffs = get_camera_intrinsics()
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
target_id = 42
marker_length = 0.1

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while not rospy.is_shutdown():
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and target_id in ids:
            index = np.where(ids == target_id)[0][0]
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[index], marker_length, camera_matrix, dist_coeffs)
            if rvec is not None and tvec is not None:
                aruco.drawDetectedMarkers(frame, corners)
                aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                print("相機位置:", tvec[0][0])
                print("相機旋轉:", rvec[0][0])
                publish_to_tf(rvec, tvec)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
