import serial
import cv2

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
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    ret, frame = cam.read()
    out.write(frame)    # Write the frame to the video file and save it

    # Example message with left and right motor speeds
    msg = {"T": 1, "L": 0.5, "R": -0.5} 


    send_serial(serial_port, str(msg))

    serial_port.close()
    cam.release()
    out.release()
    cv2.destroyAllWindows()


