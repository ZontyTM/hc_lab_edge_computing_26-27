import cv2
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

marker = aruco.generateImageMarker(aruco_dict, 0, 300)

cv2.imwrite("marker.png", marker)

cv2.imshow("ArUco Marker", marker)
cv2.waitKey(0)
cv2.destroyAllWindows()