import cv2

cap = cv2.VideoCapture(0)
cnt = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == 32:
        cnt += 1
        cv2.imwrite("calib_images/" + str(cnt) + ".jpg", frame)
        print('save image: ' + str(cnt))

cap.release()
cv2.destroyAllWindows()