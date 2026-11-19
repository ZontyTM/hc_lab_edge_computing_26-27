import argparse
from typing import Tuple
import numpy as np

import cv2

def load_camera_calibration(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera matrix K and distortion coefficients from camera_calib.npz.
    """
    with np.load(path) as data:
        try:
            camera_matrix = data["K"]
            dist_coeffs = data["dist"]
        except KeyError as exc:
            raise ValueError(
                f"Calibration file must contain 'K' and 'dist'; found {data.files}"
            ) from exc

    return camera_matrix, dist_coeffs

calib = "camera_calib.npz"
K, D = load_camera_calibration(calib)
print(K, "\n" , D)

img = cv2.imread("test.jpg")
h, w = img.shape[:2]

new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
undistorted = cv2.undistort(img, K, D, None, new_K)

cv2.imwrite("planar.jpg", undistorted)

cv2.imshow("Original", img)
cv2.imshow("Undistorted", undistorted)

cv2.waitKey(0)