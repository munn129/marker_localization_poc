import numpy as np
import cv2

marker = np.zeros((300, 300,1), dtype=np.uint8)
aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
cv2.aruco.generateImageMarker(aruco_dictionary, 1, 300, marker, 1)

if not cv2.imwrite("test.png", marker):
    raise Exception("save is failed")
