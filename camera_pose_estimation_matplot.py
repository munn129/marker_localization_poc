# camera pose estimation with aruco marker
# FOR TEST (w/o ROS)

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from marker_coordinate import marker_coordinate

## IPHONE 13 Pro Max
camera_matrix = np.array([26273.684,0,540,0,26273.684,960,0,0,1], dtype=np.float32).reshape(3,3)
## IPHONE X
camera_matrix = np.array([22950.819,0,540,0,22950.819,960,0,0,1], dtype=np.float32).reshape(3,3)
## OOO
camera_matrix = np.array([1,0,0,0,1,0,0,0,1], dtype=np.float32).reshape(3,3)
## Video 1
camera_matrix = np.array([1861.97717, 0, 960, 0, 1861.23669, 540,0,0,1]).reshape(3,3)
distort_coefficient = np.array([0,0,0,0,0], dtype=np.float32)

marker_length = 0.03


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
count_list = []
count = 0
count_time = 0

class MAKER_ESTIMATOR:
    def __init__(self):
        pass


    def estimation_video(self, video_path, ):
        cap = cv2.VideoCapture(video_path)
        #cap = cv2.VideoCapture(0)
        if cap.isOpened():
            while True:
                count_time +=1
                print(count_time)
                ret, img = cap.read()
                start = time.time()
                ret, img = cap.read()
                if count_time > 0:
                    if ret:
                        corners, ids, rejected = detector.detectMarkers(img)
                    else:
                        print('reading video has a problem')
                        break
                    s = f'detection time is {round((time.time()-start)*1000, 2)}ms'

                if ids is not None:
                    camera_pose_post = [0,0,0,0]
                    for i in range(0, len(ids)):
                        # if ids[i,0] == 7:
                        if ids[i,0] == 10:
                            _, rvec, tvec = cv2.solvePnP(object_points, corners[i], camera_matrix, distort_coefficient)
                            img = cv2.drawFrameAxes(img, camera_matrix, distort_coefficient, rvec, tvec, 0.03)
                            img = cv2.putText(img, s, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,120,120), 2)
                            if ids[i,0] < 11:
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

                                camera_pose_post = [ x + y for x, y in zip(camera_pose_post, camera_pose)]
                        if(camera_pose_post[-1] != 0):
                            camera_pose_post = [i/camera_pose_post[-1] for i in camera_pose_post]
                            camera_pose_post[-1] = 1
                            # print(camera_pose_post)

                            x_data.append(camera_pose_post[0])
                            y_data.append(camera_pose_post[1])
                            z_data.append(abs(camera_pose_post[2]))
                            # z_data.append(camera_pose_post[2])
                            # z_data.append(0)
                            count_list.append(count)
                            # x_data.append(0)
                            # y_data.append(0)
                            # z_data.append(camera_pose_post[2])
                            ax.clear()
                            
                            ax.scatter(z_data, x_data, y_data ,c = count_list, marker='o', cmap='viridis')
                            count += 1
                            ax.set_xlabel('X-directioin')
                            ax.set_ylabel('Y-directioin')
                            ax.set_zlabel('Z-directioin')
                            plt.show()
                            # plt.pause(0.01)

                cv2.imshow(video_path, img)
                cv2.waitKey(1)

        else:
            print("cannot open video file")

        cap.release()
        cv2.destroyAllWindows()

    def main(self, video_path):
        pass

if __name__ == '__main__':
    video_path = './test_images/video_3.mp4'
    estimation_marker = MAKER_ESTIMATOR()
    estimation_marker.main()