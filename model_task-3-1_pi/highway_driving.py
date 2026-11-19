"""Sample Webots controller for highway driving benchmark."""

import serial
import cv2
import threading
import numpy as np
import threading

def read_serial(serial_port):
    raw = serial_port.readline().decode('utf-8').strip()
    print(f"Received: {raw}")

def send_serial(serial_port, message):
    serial_port.write((message + '\n').encode())
    serial_port.flush()

if __name__ == "__main__":
    serial_port = serial.Serial('/dev/serial0', 115200, dsrdtr=None)
    serial_port.setRTS(False)
    serial_port.setDTR(False)

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera Frame Width: {frame_width}, Frame Height: {frame_height}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    # for i in range(0, 1000):
    #     ret, frame = cam.read()
    #     out.write(frame)    # Write the frame to the video file and save it

    src = np.float32([
        [0, 461],
        [152, 244],
        [436, 244],
        [610, 461]
    ])

    dst = np.float32([
        [0, 127],
        [0, 0],
        [255, 0],
        [255, 127]
    ])

    width = 255
    height = 127

    #cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Result", 1000, 1000)

    matrix = cv2.getPerspectiveTransform(src, dst)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 1000, 1000)

    for i in range(0, 100):
        _, img = cam.read()
        #cv2.imwrite("camera.png", img)
        #exit(0)
        #out.write(img)

        #img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        bev = cv2.warpPerspective(img, matrix, (256, 127))

        #cv2.imshow("Result", bev)
        #key = cv2.waitKey(25)

        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (9,9), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
    
        #sx = np.abs(cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3))
        #edges = np.uint8(255 * sx / (sx.max() + 1e-6))
        #_, edges = cv2.threshold(edges, 15, 255, cv2.THRESH_BINARY)
        #hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        #grass_mask = cv2.inRange(hsv, np.array([25, 40, 20]), np.array([95, 255, 255]))
        #grass_mask = cv2.dilate(grass_mask, np.ones((3, 3), np.uint8))
        #edges[grass_mask > 0] = 0

        med = np.median(denoised)
        edges = cv2.Canny(denoised, threshold1=int(0.3*med), threshold2=int(0.8*med), apertureSize=3, L2gradient=True)
        edges_denoised = cv2.GaussianBlur(edges, (9,9), 0)

        #cv2.imshow("Result", edges_denoised)
        lines = cv2.HoughLinesP(edges_denoised, rho=1, theta=np.pi / 180, threshold=90, minLineLength=40, maxLineGap=60)
        result = bev.copy()

        def get_angle(line):
            x1, y1, x2, y2 = line
            return np.degrees(np.arctan2(y2 - y1, x2 - x1))

        #result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        #cv2.imshow("Result", result)

        vehicle_center_x = 117
        vehicle_reference_y = 126.0

        left_lane_line = None
        right_lane_line = None

        left_min_dif = -1
        right_min_dif = -1

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line

                if y2 != y1:
                    slope = (x2 - x1) / (y2 - y1)
                    intercept = x1 - slope * y1
                else:
                    continue

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
                    x1, y1, x2, y2 = line
                    cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255) if i == 0 else (255, 0, 0), 3)

        steering_angle = 0
        if left_lane_line is not None and right_lane_line is not None:
            lx1, ly1, lx2, ly2 = left_lane_line
            rx1, ry1, rx2, ry2 = right_lane_line
            left_x_at_vehicle = np.polyfit([ly1, ly2], [lx1, lx2], 1)
            right_x_at_vehicle = np.polyfit([ry1, ry2], [rx1, rx2], 1)
            drive_lane_x_mid = (
                np.polyval(left_x_at_vehicle, vehicle_reference_y)
                + np.polyval(right_x_at_vehicle, vehicle_reference_y)
            ) / 2
            lateral_offset = drive_lane_x_mid - vehicle_center_x

            heading_error_degrees = (
                (get_angle(left_lane_line) % 180 + get_angle(right_lane_line) % 180) / 2
            )
            heading_error_degrees -= 90

            steering_gain_heading = 0.005
            steering_gain_lateral = 0.005
    
            steering_angle = (
                steering_gain_heading * heading_error_degrees
                + steering_gain_lateral * lateral_offset
            )

            print(
                f"Heading error: {heading_error_degrees:.2f} degrees, "
                f"Lateral offset: {lateral_offset:.2f} pixels"
            )
        
        cv2.imshow("Result", result)
        key = cv2.waitKey(1)

        steering_angle = np.clip(steering_angle, -0.5, 0.5)

        L = 0.5
        R = 0.5

        if steering_angle < 0:
            L = 0.4
        else:
            R = 0.4

        msg = {"T": 1, "L": L, "R": R}
        send_serial(serial_port, str(msg))

        #steering_angle = np.clip(steering_angle, -0.5, 0.5)
        #driver.setSteeringAngle(float(steering_angle))

        '''display_image = result #cv2.resize(result, (128, 64))
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
        maxSpeed = 80'''
    

    serial_port.close()
    cam.release()
    out.release()
    cv2.destroyAllWindows()
