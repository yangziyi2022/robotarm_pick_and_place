#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco

# 定義 ArUco 字典
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# 生成 ArUco 標籤
tag_id = 42  # 標籤的唯一 ID
tag_size = 200  # 標籤尺寸 (像素)
tag_image = aruco.drawMarker(aruco_dict, tag_id, tag_size)

# 保存為圖片
cv2.imwrite("aruco_marker.png", tag_image)

import os
print("Current Working Directory:", os.getcwd())
