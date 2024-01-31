import math
import numpy as np

marker_coordinate ={
    1 : (1,1),
    2 : (1.06, 1),
    3 : (1.12, 1),
    4 : (1.18, 1),
    5 : (1.24, 1),
    6 : (1, 0.94),
    7 : (1.06, 0.94),
    8 : (1.12, 0.94),
    9 : (1.18, 0.94),
    10 : (1.24, 0.94)
}

def homogeneous_matrix_maker(roll, pitch, yaw, x, y, z): 

    homogeneous_matrix = np.eye(4)
    yaw_mat = np.array([math.cos(yaw), -math.sin(yaw), 0, math.sin(yaw), math.cos(yaw), 0, 0, 0, 1]).reshape(3,3)
    pitch_mat = np.array([math.cos(pitch), 0, math.sin(pitch), 0, 1, 0, -math.sin(pitch), 0, math.cos(pitch)]).reshape(3,3)
    roll_mat = np.array([1,0,0,0,math.cos(roll), -math.cos(roll), 0, math.sin(roll), math.cos(roll)]).reshape(3,3)
    homogeneous_matrix[:3, :3] = yaw_mat @ pitch_mat @ roll_mat
    homogeneous_matrix[:3, 3] = np.array([x,y,z]).reshape(1,3)

    return homogeneous_matrix