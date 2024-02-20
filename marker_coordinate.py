import math
import numpy as np

marker_coordinate ={
    1 : (0,90,0,1,1,1),
    2 : (0,90,0,1.06,1,1),
    3 : (0,90,0,1.12,1,1),
<<<<<<< HEAD
    4 : (0,90,0,1.18,1,1),
    5 : (0,90,0,1.24,1,1),
    6 : (0,90,0,1, 0.94,1),
    7 : (0,90,0,1.06, 0.94,1),
    8 : (0,90,0,1.12, 0.94,1),
    9 : (0,90,0,1.18, 0.94,1),
    10 : (0,90,0,1.24, 0.94,1)
=======
    4 : (0,0,0,1.18,1,1),
    5 : (0,0,0,1.24,1,1),
    6 : (0,0,0,1, 0.94,1),
    7 : (0,0,0,1.06, 0.94,1),
    8 : (0,0,0,1.12, 0.94,1),
    9 : (0,0,0,1.18, 0.94,1),
    # 10 : (0,90,0,1.24, 0.94,1)
    10 : (0, 0, 0, 0,0,0,0)
>>>>>>> 9a65c6e0f285dc858efcec85a1aa9ff09723b915
}

def homogeneous_matrix_maker(roll, pitch, yaw, x, y, z): 
    '''
    roll, pitch, yaw : rotation vector , world coordinate(UTM) <-> Marker
    x, y, z : translation vector
    '''
    homogeneous_matrix = np.eye(4)
    yaw_mat = np.array([math.cos(yaw), -math.sin(yaw), 0, math.sin(yaw), math.cos(yaw), 0, 0, 0, 1]).reshape(3,3)
    pitch_mat = np.array([math.cos(pitch), 0, math.sin(pitch), 0, 1, 0, -math.sin(pitch), 0, math.cos(pitch)]).reshape(3,3)
    roll_mat = np.array([1,0,0,0,math.cos(roll), -math.cos(roll), 0, math.sin(roll), math.cos(roll)]).reshape(3,3)
    homogeneous_matrix[:3, :3] = (yaw_mat @ pitch_mat) @ roll_mat
    homogeneous_matrix[:3, 3] = np.array([x,y,z]).reshape(1,3)
    homogeneous_matrix[3,3] = 1

    return homogeneous_matrix

# in porgress
def get_homogeneous_matrix(id):
    roll = marker_coordinate[id][0] * math.pi / 180
    pitch = marker_coordinate[id][1] * math.pi / 180
    yaw = marker_coordinate[id][2] * math.pi / 180
    x = marker_coordinate[id][3]
    y = marker_coordinate[id][4]
    z = marker_coordinate[id][5]
    
    return homogeneous_matrix_maker(roll, pitch, yaw, x, y, z)
