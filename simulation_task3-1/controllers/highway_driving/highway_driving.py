"""Sample Webots controller for highway driving benchmark."""

from vehicle import Driver
import cv2
import threading
import numpy as np
from controller import Display
import threading

if __name__ == "__main__":
    # get the camera
    driver = Driver()
    driver.setSteeringAngle(0.0)  # go straight

    camera = driver.getDevice('camera')
    camera.enable(50)
    #camera.recognitionEnable(50)
    display = driver.getDevice("display")

    src = np.float32([
        [5, 99],
        [73, 80],
        [194, 80],
        [255, 99]
    ])

    diff = np.ptp(src, axis=0).astype(int)

    dst = np.float32([
        [0, 127],
        [0, 0],
        [255, 0],
        [255, 127]
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)

    while driver.step() != -1:

        driver.setSteeringAngle(0.0)
        driver.setCruisingSpeed(20)


        image = camera.getImage()

        width = camera.getWidth()
        height = camera.getHeight()


        img = np.frombuffer(image, dtype=np.uint8)
        img = img.reshape((height, width, 4))

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        bev = cv2.warpPerspective(img, matrix, (256, 127))

        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (3,3), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
    
        sx = np.abs(cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3))
        edges = np.uint8(255 * sx / (sx.max() + 1e-6))
        _, edges = cv2.threshold(edges, 15, 255, cv2.THRESH_BINARY)

        hough_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=90, minLineLength=40, maxLineGap=60)
        result = bev.copy()
        lines = []

        def get_angle(line):
            x1, y1, x2, y2 = line[0]
            return np.degrees(np.arctan2(y2 - y1, x2 - x1))

        result = bev.copy()
        if hough_lines is not None:
            for line in hough_lines:
                angle = get_angle(line)

                # Nur nahezu senkrechte Linien (+/- 5 Grad)
                if abs(abs(angle) - 90) <= 20:
                    lines.append(line)

        vehicle_center_x = 127
        vehicle_reference_y = 126.0

        left_lane_line = None
        right_lane_line = None

        left_min_dif = -1
        right_min_dif = -1

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Fit x = slope * y + intercept
                slope, intercept = np.polyfit(
                    [y1, y2],
                    [x1, x2],
                    1
                )

                x_at_vehicle = slope * vehicle_reference_y + intercept

                if x_at_vehicle < vehicle_center_x:
                    dif = vehicle_center_x - x_at_vehicle
                    if (left_min_dif := dif if left_min_dif == -1 else min(dif, left_min_dif)) == dif:
                        left_lane_line = line

                    #left_lane_lines.append((slope, intercept))
                    cv2.line(result, (x1, y1), (x2, y2), (0, 0, 0), 3)
                else:
                    dif = x_at_vehicle - vehicle_center_x
                    if (right_min_dif := dif if right_min_dif == -1 else min(dif, right_min_dif)) == dif:
                        right_lane_line = line
                        
                    #right_lane_lines.append((slope, intercept))
                    cv2.line(result, (x1, y1), (x2, y2), (0, 0, 0), 3)


            for i, line in enumerate([left_lane_line, right_lane_line]):
                if line is not None:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255) if i == 0 else (255, 0, 0), 3)

        steering_angle = 0
        if left_lane_line is not None and right_lane_line is not None:
            lx1, ly1, lx2, ly2 = left_lane_line[0]
            rx1, ry1, rx2, ry2 = right_lane_line[0]
            drive_lane_x_mid = lx1 + (rx1 - lx1) / 2
            lateral_offset = drive_lane_x_mid - vehicle_center_x

            heading_error_degrees = (get_angle(left_lane_line) + get_angle(right_lane_line)) / 2
            #heading_error_degrees = get_angle(right_lane_line)
            #print("PRE: ", heading_error_degrees)
            heading_error_degrees -= 90
            #print("POST: ", heading_error_degrees)

            steering_gain_heading = 0.02
            steering_gain_lateral = 0.005
    
            steering_angle = (
                steering_gain_heading * heading_error_degrees
                + steering_gain_lateral * lateral_offset
            )

            print(
                f"Heading error: {heading_error_degrees:.2f} degrees, "
                f"Lateral offset: {lateral_offset:.2f} pixels"
            )



        # Calculate lane centerline only when both lane boundaries are detected
        '''if len(left_lane_lines) > 0 and len(right_lane_lines) > 0:
            left_slope, left_intercept = np.mean(left_lane_lines, axis=0)
            right_slope, right_intercept = np.mean(right_lane_lines, axis=0)

            center_slope = (left_slope + right_slope) / 2.0
            center_intercept = (left_intercept + right_intercept) / 2.0

            # Lane center position at vehicle position
            lane_center_x = (
                center_slope * vehicle_reference_y
                + center_intercept
            )

            # Signed lateral offset in pixels
            lateral_offset = lane_center_x - vehicle_center_x

            # Lane direction farther ahead in the bird's-eye image
            forward_y = 20.0

            center_x_ahead = (
                center_slope * forward_y
                + center_intercept
            )

            delta_x = center_x_ahead - lane_center_x
            delta_y = vehicle_reference_y - forward_y

            # Positive value means the lane direction points to the right
            heading_error = np.arctan2(delta_x, delta_y)

            heading_error_degrees = np.degrees(heading_error)

            print(
                f"Heading error: {heading_error_degrees:.2f} degrees, "
                f"Lateral offset: {lateral_offset:.2f} pixels"
            )

            # Draw estimated lane centerline
            center_x_bottom = int(lane_center_x)
            center_x_top = int(center_x_ahead)

            cv2.line(
                result,
                (center_x_bottom, int(vehicle_reference_y)),
                (center_x_top, int(forward_y)),
                (0, 255, 0),
                3
            )
        else:
            heading_error_degrees = 0.0
            lateral_offset = 0.0


        steering_gain_heading = 0.02
        steering_gain_lateral = 0.005

        steering_angle = (
            steering_gain_heading * heading_error_degrees
            + steering_gain_lateral * lateral_offset
        )'''

        steering_angle = np.clip(steering_angle, -0.5, 0.5)
        driver.setSteeringAngle(float(steering_angle))

        '''if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)'''

        #cv2.imwrite("/home/team1/hc_lab_edge_computing_26-27/camera.png", img)
        #cv2.imwrite("/home/team1/hc_lab_edge_computing_26-27/camera_after_computation.png", result)
        #break
        display_image = result #cv2.resize(result, (128, 64))
        display_bgra = cv2.cvtColor(display_image, cv2.COLOR_BGR2BGRA)

        image_ref = display.imageNew(
            display_bgra.tobytes(),
            Display.BGRA,
            256,
            128
        )

        if image_ref:
            display.imagePaste(image_ref, 0, 0, False)
            display.imageDelete(image_ref)            
        maxSpeed = 80
