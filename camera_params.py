import numpy as np

default_camera_mat = np.array([1,0,0,0,1,0,0,0,1], dtype=np.float32).reshape(3,3)
iphone13_camera_mat = np.array([26273.684,0,540,0,26273.684,960,0,0,1], dtype=np.float32).reshape(3,3)
iphoneX_camera_mat = np.array([22950.819,0,540,0,22950.819,960,0,0,1], dtype=np.float32).reshape(3,3)
sekonix_camera_mat = camera_matrix = np.array([1861.97717, 0, 982.67999, 0, 1861.23669, 541.22597,0,0,1]).reshape(3,3)

default_distort_coefficient = np.array([0,0,0,0,0], dtype=np.float32)
sekonix_distort_coefficient = np.array([-0.33022, 0.31466, -0.00036, -0.00093, -0.34203])