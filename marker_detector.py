import cv2

img = cv2.imread('test_cap.png', cv2.IMREAD_COLOR)
output_img = img[:]

# ids = 1
# corners = []
# rejected = []

detector_parameter = cv2.aruco.DetectorParameters()
marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

detector = cv2.aruco.ArucoDetector(marker_dictionary, detector_parameter)
corners, ids, rejected = detector.detectMarkers(img)
cv2.aruco.drawDetectedMarkers(output_img, corners, ids)

if not cv2.imwrite("detect_output.png", output_img):
    raise Exception("save is failed")