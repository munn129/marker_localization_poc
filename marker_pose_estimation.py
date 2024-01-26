import cv2
import numpy as np

#camera_matrix = [1,0,0,0,1,0,0,0,1] # f_x, 0, c_x, 0, f_y, c_y, 0,0,1
#camera_matrix = np.array([1,0,2016,0,1,1512,0,0,1], dtype=np.float32).reshape(3,3)
#distort_coefficient = np.array([0,0,0,0], dtype=np.float32)

# iphone camera intrinsic parameter
camera_matrix = np.array([[1.97547873e+03, 0.00000000e+00, 1.06077279e+03],
                          [0.00000000e+00, 2.05341424e+03, 5.13500761e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
distort_coefficient = np.array([4.189866e-01, -3.496832e+00, -1.239132e-02, -1.649367e-03, 1.055315e+01], dtype=np.float32)

marker_length = 0.03 #3cm?

img = cv2.imread('test_images/marker.jpg', cv2.IMREAD_COLOR)
output_img = img[:]

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

corners, ids, rejected = detector.detectMarkers(img)

if len(ids) > 0:
    cv2.aruco.drawDetectedMarkers(output_img, corners, ids)

    rotation_vector = []
    translation_vector = []

    for i in range(0, len(ids)):
        _, rvec, tvec = cv2.solvePnP(object_points, corners[i], camera_matrix, distort_coefficient)
        rotation_vector.append(rvec)
        translation_vector.append(tvec)

    for i in range(0, len(ids)):
        output_img = cv2.drawFrameAxes(img, camera_matrix, distort_coefficient, rotation_vector[i], translation_vector[i], 0.03)
    
if not cv2.imwrite("iphone.png", output_img):
    raise Exception("save is failed")
