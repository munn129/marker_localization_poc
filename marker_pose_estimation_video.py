import cv2
import numpy as np
from marker_coordinate import marker_coordinate

camera_matrix = np.array([1,0,640,0,1,360,0,0,1], dtype=np.float32).reshape(3,3)
distort_coefficient = np.array([0,0,0,0], dtype=np.float32)

marker_length = 0.03

video = './test_video.mp4'

# set coordinate system
object_points = np.zeros((4, 3), dtype=np.float32)

object_points[0] = np.array([-marker_length/2.0, marker_length/2.0, 0], dtype=np.float32)
object_points[1] = np.array([marker_length/2.0, marker_length/2.0, 0], dtype=np.float32)
object_points[2] = np.array([marker_length/2.0, -marker_length/2.0, 0], dtype=np.float32)
object_points[3] = np.array([-marker_length/2.0, -marker_length/2.0, 0], dtype=np.float32)

# detect marker 
detector_parameter = cv2.aruco.DetectorParameters()
marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
detector = cv2.aruco.ArucoDetector(marker_dictionary, detector_parameter)

cap = cv2.VideoCapture(video)
#cap = cv2.VideoCapture(0)
if cap.isOpened():

    while True:
        ret, img = cap.read()
        if ret:
            corners, ids, rejected = detector.detectMarkers(img)
        else:
            print('reading video has a problem')
            break

        if ids is not None:
            for i in range(0, len(ids)):
                _, rvec, tvec = cv2.solvePnP(object_points, corners[i], camera_matrix, distort_coefficient)
                img = cv2.drawFrameAxes(img, camera_matrix, distort_coefficient, rvec, tvec, 0.03)
                if ids[i,0] < 11:
                    print(marker_coordinate[ids[i,0]][0])

        cv2.imshow(video, img)
        cv2.waitKey(33)

else:
    print("cannot open video file")

cap.release()
cv2.destroyAllWindows()
