#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs

def nothing(x):
    pass

# 初始化 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 創建 HSV 滑桿窗口 用黃色方塊示範
cv2.namedWindow("HSV Adjustment")
cv2.createTrackbar("H Lower", "HSV Adjustment", 20, 180, nothing)
cv2.createTrackbar("H Upper", "HSV Adjustment", 30, 180, nothing)
cv2.createTrackbar("S Lower", "HSV Adjustment", 100, 255, nothing)
cv2.createTrackbar("S Upper", "HSV Adjustment", 255, 255, nothing)
cv2.createTrackbar("V Lower", "HSV Adjustment", 67, 255, nothing)
cv2.createTrackbar("V Upper", "HSV Adjustment", 255, 255, nothing)

try:
    while True:
        # 獲取相機幀數據
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 獲取 RGB 圖像
        color_image = np.asanyarray(color_frame.get_data())
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # 讀取滑桿的 HSV 範圍 
        h_lower = cv2.getTrackbarPos("H Lower", "HSV Adjustment")
        h_upper = cv2.getTrackbarPos("H Upper", "HSV Adjustment")
        s_lower = cv2.getTrackbarPos("S Lower", "HSV Adjustment")
        s_upper = cv2.getTrackbarPos("S Upper", "HSV Adjustment")
        v_lower = cv2.getTrackbarPos("V Lower", "HSV Adjustment")
        v_upper = cv2.getTrackbarPos("V Upper", "HSV Adjustment")

        # 設定 HSV 範圍
        lower_hsv = np.array([h_lower, s_lower, v_lower])
        upper_hsv = np.array([h_upper, s_upper, v_upper])

        # 創建掩膜圖
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 顯示圖像
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Mask", mask)

        # 按 ESC 鍵退出
        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
