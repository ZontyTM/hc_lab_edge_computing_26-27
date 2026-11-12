import cv2
import numpy as np

img = cv2.imread("camera.png")

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 1000, 1000)

'''src = np.float32([
    [5, 99],
    [73, 80],
    [194, 80],
    [255, 99]
])'''

src = np.float32([
    [0, 127],
    [105, 75],
    [150, 75],
    [255, 127]
])

#diff = np.ptp(src, axis=0).astype(int)

dst = np.float32([
    [0, 127],
    [0, 0],
    [255, 0],
    [255, 127]
])

matrix = cv2.getPerspectiveTransform(src, dst)
bev = cv2.warpPerspective(img, matrix, (256, 127))

gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
denoised = cv2.GaussianBlur(gray, (3,3), 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(denoised)


#med = np.median(denoised)
#edges = cv2.Canny(denoised, threshold1=int(0.3*med), threshold2=int(0.8*med), apertureSize=3, L2gradient=True)
#sx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
#sy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

#sobel = cv2.magnitude(sx, sy)
#cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#_, edges = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)

def getlines():
    sx = np.abs(cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3))
    edges = np.uint8(255 * sx / (sx.max() + 1e-6))
    _, edges = cv2.threshold(edges, 15, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Result", edges)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=40, maxLineGap=60)
    vertical_lines = []

    result = enhanced.copy()
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Nur nahezu senkrechte Linien (+/- 5 Grad)
            if abs(abs(angle) - 90) <= 5:
                vertical_lines.append(line)
                overlay = np.zeros_like(result)

                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
                alpha = 0.2
                result = cv2.addWeighted(overlay, alpha, result, 1, 0)

    return result

#result = img
result = getlines()

cv2.imshow("Result", result)

cv2.waitKey(0)

cv2.destroyAllWindows()

