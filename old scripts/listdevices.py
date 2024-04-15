import cv2

def list_devices():
    devices = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            devices.append(i)
            cap.release()
    return devices

available_devices = list_devices()
print("Available devices:", available_devices)