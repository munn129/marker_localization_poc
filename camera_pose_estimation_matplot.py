# camera pose estimation with aruco marker
# FOR TEST (w/o ROS)

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from marker_coordinate import marker_coordinate

# camera_matrix = np.array([26273.684,0,540,0,26273.684,960,0,0,1], dtype=np.float32).reshape(3,3)
# camera_matrix = np.array([22950.819,0,540,0,22950.819,960,0,0,1], dtype=np.float32).reshape(3,3)
# camera_matrix = np.array([1,0,0,0,1,0,0,0,1], dtype=np.float32).reshape(3,3)
# camera_matrix = np.array([1861.97717, 0, 960, 0, 1861.23669, 540,0,0,1]).reshape(3,3)
camera_matrix = np.array([18610, 0, 960, 0, 1861, 5400,0,0,1]).reshape(3,3)
distort_coefficient = np.array([0,0,0,0,0], dtype=np.float32)

marker_length = 30

video = './test_images/video_3.mp4'

# set coordinate system
object_points = np.zeros((4, 3), dtype=np.float32)

object_points[0] = np.array([-marker_length/2.0, marker_length/2.0, 0], dtype=np.float32)
object_points[1] = np.array([marker_length/2.0, marker_length/2.0, 0], dtype=np.float32)
object_points[2] = np.array([marker_length/2.0, -marker_length/2.0, 0], dtype=np.float32)
object_points[3] = np.array([-marker_length/2.0, -marker_length/2.0, 0], dtype=np.float32)
camera_matrix = np.array([1,0,0,0,1,0,0,0,1], dtype=np.float32).reshape(3,3)

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
            camera_pose_post = [0,0,0,0]
            for i in range(0, len(ids)):
                _, rvec, tvec = cv2.solvePnP(object_points, corners[i], camera_matrix, distort_coefficient)
                img = cv2.drawFrameAxes(img, camera_matrix, distort_coefficient, rvec, tvec, 30)

                if ids[i,0] < 11:
                # if ids[i,0] == 1:
                    rotation_matrix = Rotation.from_euler('xyz', rvec.reshape(1,3), degrees=False).as_matrix()
                    pose_matrix = np.eye(4)
                    pose_matrix[:3, :3] = rotation_matrix
                    pose_matrix[:3, 3] = tvec.reshape(1,3)

                    marker_x = marker_coordinate[ids[i,0]][0]
                    marker_y = marker_coordinate[ids[i,0]][1]
                    # marker_z = marker_coordinate[ids[i,0]][2]
                    marker_z = 0

                    marker_absolute_position = np.array([marker_x, marker_y, marker_z, 1])
                    camera_pose = np.linalg.inv(pose_matrix) @ marker_absolute_position

                    # for average camera pose
                    camera_pose_post = [ x + y for x, y in zip(camera_pose_post, camera_pose)]
                    
            if(camera_pose_post[-1] != 0):
                # average camera pose estimated from each marker
                # absolute camera pose |value|
                camera_pose_post = [i for i in camera_pose_post]
                camera_pose_post[-1] = 1
                # print(camera_pose_post)

                # x_data.append(abs(camera_pose_post[0]))
                # y_data.append(abs(camera_pose_post[1]))
                # z_data.append(abs(camera_pose_post[2]))
                x_data.append(camera_pose_post[0])
                y_data.append(camera_pose_post[1])
                z_data.append(camera_pose_post[2])
                ax.clear()
                ax.scatter(x_data, y_data, z_data, marker='o')
                # ax.plot(x_data, y_data, z_data)

                plt.xlabel('X [mm]')
                plt.ylabel('Y [mm]')
                plt.show()
                plt.pause(0.01)

        cv2.imshow(video, img)
        cv2.waitKey(33)

else:
    print("cannot open video file")

cap.release()
cv2.destroyAllWindows()
