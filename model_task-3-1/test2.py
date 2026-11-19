import cv2
import numpy as np

'''src = np.float32([
    [94, 717],
    [267, 475],
    [783, 464],
    [941, 718]
])'''

'''src = np.float32([
    [0, 479],
    [117, 304],
    [452, 297],
    [569, 479]
])'''

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

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 1000, 1000)

img = cv2.imread("camera3.png")
matrix = cv2.getPerspectiveTransform(src, dst)
bev = cv2.warpPerspective(img, matrix, (256, 127))

gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
denoised = cv2.GaussianBlur(gray, (9,9), 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(denoised)

def getlines():
    #sx = np.abs(cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=31))
    #edges = np.uint8(255 * sx / (sx.max() + 1e-6))
    #_, edges = cv2.threshold(edges, 15, 255, cv2.THRESH_BINARY)
    med = np.median(denoised)
    edges = cv2.Canny(denoised, threshold1=int(0.3*med), threshold2=int(0.8*med), apertureSize=3, L2gradient=True)
    edges_denoised = cv2.GaussianBlur(edges, (9,9), 0)

    #hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
    #grass_mask = cv2.inRange(hsv, np.array([25, 40, 20]), np.array([95, 255, 255]))
    #grass_mask = cv2.dilate(grass_mask, np.ones((3, 3), np.uint8))
    #edges[grass_mask > 0] = 0
    #cv2.imshow("Result", edges_denoised)

    lines = cv2.HoughLinesP(edges_denoised, rho=1, theta=np.pi / 180, threshold=90, minLineLength=40, maxLineGap=60)
    vertical_lines = []

    result = enhanced.copy()
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Nur nahezu senkrechte Linien (+/- 5 Grad)
            if abs(abs(angle) - 90) <= 90:
                vertical_lines.append(line)
                overlay = np.zeros_like(result)

                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
                alpha = 0.3
                result = cv2.addWeighted(overlay, alpha, result, 1, 0)

    return result

#result = img
result = getlines()

cv2.imshow("Result", result)

cv2.waitKey(0)

cv2.destroyAllWindows()

# 94, 717
# 267, 475
# 783, 464
# 941, 718