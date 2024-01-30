# camera pose estimation with aruco marker
# FOR TEST (w/o ROS)

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from marker_coordinate import marker_coordinate

class MAKER_ESTIMATOR:
    def __init__(self, marker_length, marker_type, flag, input, num_marker):
        self.input = input
        self.num_marker = num_marker

        # detect marker 
        detector_parameter = cv2.aruco.DetectorParameters()
        marker_dictionary = cv2.aruco.getPredefinedDictionary(marker_type)
        self.detector = cv2.aruco.ArucoDetector(marker_dictionary, detector_parameter)
        
        # set coordinate system
        self.object_points = np.zeros((4, 3), dtype=np.float32)

        self.object_points[0] = np.array([-marker_length/2.0, marker_length/2.0, 0], dtype=np.float32)
        self.object_points[1] = np.array([marker_length/2.0, marker_length/2.0, 0], dtype=np.float32)
        self.object_points[2] = np.array([marker_length/2.0, -marker_length/2.0, 0], dtype=np.float32)
        self.object_points[3] = np.array([-marker_length/2.0, -marker_length/2.0, 0], dtype=np.float32)

        if flag == 'default':
            self.camera_mat = np.array([1,0,0,0,1,0,0,0,1], dtype=np.float32).reshape(3,3)
            self.distort_coefficient = np.array([0,0,0,0,0], dtype=np.float32)
        if flag == 'iphone13':
            self.camera_mat = np.array([26273.684,0,540,0,26273.684,960,0,0,1], dtype=np.float32).reshape(3,3)
            self.distort_coefficient = np.array([0,0,0,0,0], dtype=np.float32)
        if flag == 'iphonex':
            self.camera_mat = np.array([22950.819,0,540,0,22950.819,960,0,0,1], dtype=np.float32).reshape(3,3)
            self.distort_coefficient = np.array([0,0,0,0,0], dtype=np.float32)
        if flag == 'sekonix':
            self.camera_mat = np.array([1861.97717, 0, 982.67999, 0, 1861.23669, 541.22597,0,0,1]).reshape(3,3)
            self.distort_coefficient = np.array([-0.33022, 0.31466, -0.00036, -0.00093, -0.34203])


        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.count_list = []
        self.count = 0
        self.count_time = 0
        self.one_marker = None
        self.save_marker = False

    def plot_img(self, camera_pose_post):
        # initialize graph

        if(camera_pose_post[-1] != 0):
            camera_pose_post = [i/camera_pose_post[-1] for i in camera_pose_post]
            camera_pose_post[-1] = 1
            # print(camera_pose_post)

            self.x_data.append(camera_pose_post[0])
            self.y_data.append(camera_pose_post[1])
            self.z_data.append(abs(camera_pose_post[2]))
            # z_data.append(camera_pose_post[2])
            # z_data.append(0)
            self.count_list.append(self.count)
            # x_data.append(0)
            # y_data.append(0)
            # z_data.append(camera_pose_post[2])
            self.ax.clear()
            
            self.ax.scatter(self.z_data, self.x_data, self.y_data ,c = self.count_list, marker='o', cmap='viridis')
            self.count += 1
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            plt.show()
            # plt.pause(0.01)

    def draw_img(self, img, marker_idx, corner, s_txt, camera_pose_post):

        _, rvec, tvec = cv2.solvePnP(self.object_points, corner, self.camera_mat, self.distort_coefficient)
        img = cv2.drawFrameAxes(img, self.camera_mat, self.distort_coefficient, rvec, tvec, 0.03)
        img = cv2.putText(img, s_txt, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,215,255), 2)
        if marker_idx < 11:
            rotation_matrix = Rotation.from_euler('xyz', rvec.reshape(1,3), degrees=True).as_matrix()
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rotation_matrix
            pose_matrix[:3, 3] = tvec.reshape(1,3)

            marker_x = marker_coordinate[marker_idx][0]
            marker_y = marker_coordinate[marker_idx][1]
            # marker_z = marker_coordinate[marker_idx][2]
            marker_z = 0

            marker_absolute_position = np.array([marker_x, marker_y, marker_z, 1])
            camera_pose = np.linalg.inv(pose_matrix) @ marker_absolute_position

            camera_pose_post = [ x + y for x, y in zip(camera_pose_post, camera_pose)]
        self.plot_img(camera_pose_post)

    def estimation_video(self, video_path, ):
        plt.ion()
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

        if self.input == 'video':
            cap = cv2.VideoCapture(video_path)
        if self.input == 'web_cam':
            cap = cv2.VideoCapture(0)
        if cap.isOpened():
            while True:
                self.count_time +=1
                print(f'FRAM is : {self.count_time}')
                ret, img = cap.read()
                start = time.time()
                print(f'IMG size is : {img.shape}')
                if self.count_time > 0:
                    if ret:
                        corners, ids, rejected = self.detector.detectMarkers(img)
                    else:
                        print('reading video has a problem')
                        break
                    if self.num_marker == 'single' and not self.save_marker:
                        self.one_marker = ids[0,0]
                        self.save_marker = True
                    s_txt = f'detection time is {round((time.time()-start)*1000, 2)}ms id is {self.one_marker}'


                    if ids is not None:
                        camera_pose_post = [0,0,0,0]
                        for i in range(0, len(ids)):
                            marker_idx = ids[i,0]
                            corner = corners[i]
                            if marker_idx == self.one_marker:
                                self.draw_img(img, marker_idx, corner, s_txt, camera_pose_post)

                cv2.imshow(video_path, img)
                cv2.waitKey(1)

        else:
            print("cannot open video file")

        cap.release()
        cv2.destroyAllWindows()

    def main(self, video_path):
        self.estimation_video(video_path)

if __name__ == '__main__':
    flag_list = ['iphonex','iphone13','sekonix','default']
    input_list = ['web_cam', 'video']
    num_marker_list = ['single', 'multi']
    num_marker = num_marker_list[0]
    flag = flag_list[2]
    input = input_list[0]
    marker_length = 0.03
    marker_type = cv2.aruco.DICT_4X4_1000
    estimation_marker = MAKER_ESTIMATOR(marker_length, marker_type, flag, input, num_marker)

    video_path = './test_video/raw_video.mp4'

    estimation_marker.main(video_path)