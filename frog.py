# camera pose estimation with aruco marker
# FOR TEST (w/o ROS)

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from marker_coordinate import get_homogeneous_matrix, homogeneous_matrix_maker
from camera_params import default_camera_mat, iphone13_camera_mat, iphoneX_camera_mat, sekonix_camera_mat, default_distort_coefficient, sekonix_distort_coefficient

camera_matrix = iphoneX_camera_mat
distort_coefficient = default_distort_coefficient

marker_length = 0.03

video = './test_images/m1.mp4'

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

# initialize graph
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_data = []
y_data = []
z_data = []

cap = cv2.VideoCapture(video)
# cap = cv2.VideoCapture(0)
if cap.isOpened():

    while True:
        ret, img = cap.read()
        if ret:
            corners, ids, rejected = detector.detectMarkers(img)
            cv2.aruco.drawDetectedMarkers(img, corners)
        else:
            print('reading video has a problem')
            break

        if ids is not None:
            camera_pose_post = [0,0,0,0]
            for i in range(0, len(ids)):
                _, rvec, tvec = cv2.solvePnP(object_points, corners[i], camera_matrix, distort_coefficient)
                # rvec, tvec, object_points = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_length, camera_matrix, distort_coefficient)
                img = cv2.drawFrameAxes(img, camera_matrix, distort_coefficient, rvec, tvec, marker_length)

                if ids[i,0] < 11:
                # if ids[i,0] == 10:
                    
                    pose_matrix = homogeneous_matrix_maker(rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2])
                    camera_pose_h = get_homogeneous_matrix(ids[i,0]) @ np.linalg.inv(pose_matrix)
                    # camera_pose_h = np.linalg.inv(pose_matrix) @ get_homogeneous_matrix(ids[i,0])


                    # for average camera pose
                    # x_data.append(np.linalg.inv(pose_matrix)[0][3])
                    # y_data.append(np.linalg.inv(pose_matrix)[1][3])
                    # z_data.append(abs(np.linalg.inv(pose_matrix)[2][3]))
                    x_data.append(pose_matrix[0][3])
                    y_data.append(pose_matrix[1][3])
                    z_data.append(pose_matrix[2][3])

                    ax.clear()
                    ax.scatter(x_data, y_data, z_data, marker='o')

                    ax.set_xlabel('X [m]')
                    ax.set_ylabel('Y [m]')
                    ax.set_zlabel('Z [m]')

                    plt.show()
                    plt.pause(0.01)

        cv2.imshow(video, img)
        cv2.waitKey(27)

else:
    print("cannot open video file")

cap.release()
cv2.destroyAllWindows()
