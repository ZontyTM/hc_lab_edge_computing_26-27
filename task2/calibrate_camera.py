"""
Camera calibration using a printed checkerboard.
This script computes the camera intrinsics needed for pose estimation
in Task 3:
  - camera matrix K
  - distortion coefficients d

After running this script, you will obtain:
  - camera_calib.npz  (saved to disk)
  - printed values of K, d, and RMS reprojection error in the console

In Task 3, you MUST load:
  - K  (camera matrix)
  - d  (distortion coefficients)
and pass them to solvePnP.
"""

import cv2
import numpy as np
import glob
import os

# === CHECKERBOARD SETTINGS ===
PATTERN_SIZE = (8, 12)      # number of INNER corners (x, y)
SQUARE_SIZE_M = 0.025      # square size in meters (25 mm)

# === INPUT / OUTPUT ===
IMAGES_DIR = "calib_images"        # folder with captured calibration images
OUT_FILE = "camera_calib.npz"      # output file with calibration parameters


def main():
    # ------------------------------------------------------------
    # Step 1: Define 3D points of the checkerboard in world coords
    # ------------------------------------------------------------
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    # Lists to store 3D-2D correspondences
    objpoints = []   # 3D points in world coordinates
    imgpoints = []   # 2D points in image pixels

    # ------------------------------------------------------------
    # Step 2: Load calibration images
    # ------------------------------------------------------------
    images = sorted(
        glob.glob(os.path.join(IMAGES_DIR, "*.png")) +
        glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
    )

    if not images:
        raise RuntimeError(f"No calibration images found in '{IMAGES_DIR}/'")

    image_size = None
    used_images = 0

    # Criteria for sub-pixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    # ------------------------------------------------------------
    # Step 3: Detect checkerboard corners in each image
    # ------------------------------------------------------------
    for path in images:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save image size once 
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)

        found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE)
        if not found:
            # Image is skipped if corners are not detected
            continue

        # Refine corner positions to sub-pixel accuracy
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners)
        used_images += 1

    if used_images < 8:
        raise RuntimeError(
            f"Not enough valid images ({used_images}). "
            "Capture ~15–30 images with different views."
        )

    # ------------------------------------------------------------
    # Step 4: Run camera calibration
    # ------------------------------------------------------------
    # This estimates:
    #   - K    (camera matrix)
    #   - dist (distortion coefficients)
    # and computes the RMS reprojection error.
    rms, K, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # ------------------------------------------------------------
    # Step 5: Save and report results
    # ------------------------------------------------------------
    np.savez(
        OUT_FILE,
        K=K,
        dist=dist,
        rms=float(rms),
        image_size=np.array(image_size, dtype=np.int32)
    )

    print("=== Camera calibration finished ===")
    print("Saved file:", OUT_FILE)
    print("RMS reprojection error:", rms)
    print("Camera matrix K:\n", K)
    print("Distortion coefficients d:", dist.ravel())


if __name__ == "__main__":
    main()
