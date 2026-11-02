"""
Task 2 Skeleton: Speed Estimation with ArUco Markers
====================================================

YOU MUST IMPLEMENT:
- loading camera calibration (K, dist)
- loading marker map from JSON
- speed computation from position and time
- smoothing of speed
- handling missing marker detections
- latency measurement and reporting


Press ESC to exit.
"""

import argparse
import time
from typing import Dict, Tuple, Optional

import cv2
import cv2.aruco as aruco
import numpy as np


# ==================== PROVIDED HELPERS ====================

def marker_object_corners_world(x: float, y: float, s: float) -> np.ndarray:
    """
    Marker corners in WORLD coordinates (z = 0).
    Order: top-left, top-right, bottom-right, bottom-left
    """
    h = s / 2.0
    return np.array([
        [x - h, y + h, 0.0],
        [x + h, y + h, 0.0],
        [x + h, y - h, 0.0],
        [x - h, y - h, 0.0],
    ], dtype=np.float32)


# ==================== TODO FUNCTIONS====================

def load_camera_calibration(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO:
    Load camera matrix K and distortion coefficients from camera_calib.npz.
    """
    raise NotImplementedError


def load_marker_map(path: str) -> Tuple[float, Dict[int, Tuple[float, float]]]:
    """
    TODO:
    Load marker length and marker positions from marker_map.json.

    Return:
      marker_length (float)
      marker_map: dict[int] -> (x, y)
    """
    raise NotImplementedError


# ==================== MAIN ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--calib", type=str, default="camera_calib.npz")
    ap.add_argument("--marker-map", type=str, default="marker_map.json")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)

    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        return

    # ------------------------------------------------------------------
    # TODO: Load calibration and marker map
    # ------------------------------------------------------------------

    # camera_matrix, dist_coeffs = ...
    # marker_length, marker_map = ...

    # ------------------------------------------------------------------
    # ArUco setup
    # ------------------------------------------------------------------
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

    # ------------------------------------------------------------------
    # STATE 
    # ------------------------------------------------------------------
    last_position: Optional[np.ndarray] = None
    last_time: Optional[float] = None

    while True:
        t_frame_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        obj_points = []
        img_points = []

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                mid = int(mid)
                if mid not in marker_map:
                    continue

                x, y = marker_map[mid]
                obj_points.append(
                    marker_object_corners_world(x, y, marker_length)
                )
                img_points.append(
                    corners[i].reshape(4, 2).astype(np.float32)
                )

        cam_xy: Optional[np.ndarray] = None

        # ------------------------------------------------------------------
        # Pose estimation
        # ------------------------------------------------------------------
        if obj_points:
            obj_points = np.concatenate(obj_points)
            img_points = np.concatenate(img_points)

            ok, rvec, tvec = cv2.solvePnP(
                obj_points, img_points,
                camera_matrix, dist_coeffs
            )

            if ok:
                R, _ = cv2.Rodrigues(rvec)
                cam_xy = (-R.T @ tvec).reshape(3)[:2]

                cv2.drawFrameAxes(
                    frame, camera_matrix, dist_coeffs,
                    rvec, tvec, 0.08
                )

        # ==================================================================
        # TODO:
        #
        # - compute speed from cam_xy and time
        # - smooth speed
        # - handle missing marker detections
        # - measure latency
        # - report metrics
        # ==================================================================

        cv2.imshow("Task 3: Speed Estimation", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
