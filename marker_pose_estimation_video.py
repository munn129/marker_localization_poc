import cv2
import numpy as np

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
if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            output_img = img[:]
            corners, ids, rejected = detector.detectMarkers(img)
            if ids is not None:
                rotation_vector = []
                translation_vector = []

                for i in range(0, len(ids)):
                    _, rvec, tvec = cv2.solvePnP(object_points, corners[i], camera_matrix, distort_coefficient)
                    rotation_vector.append(rvec)
                    translation_vector.append(tvec)
                    output_img = cv2.drawFrameAxes(img, camera_matrix, distort_coefficient, rotation_vector[i], translation_vector[i], 0.03)

                cv2.imshow(video, output_img)
                cv2.waitKey(33)
        else:
            break
else:
    print("cannot open video file")

cap.release()
cv2.destroyAllWindows()
