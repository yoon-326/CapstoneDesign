import cv2
import time
from picamera2 import Picamera2

def init_camera1():
    """Picamera2 초기화"""
    picam2 = Picamera2(camera_num=0)
    config = picam2.create_video_configuration(
        main={"size": (1640, 1232), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.4)
    print("Camera initialized01")
    return picam2

def init_camera2():
    """Picamera2 초기화"""
    picam2 = Picamera2(camera_num=1)
    config = picam2.create_video_configuration(
        main={"size": (1270, 720), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.4)
    print("Camera initialized02")
    return picam2


def get_frame(picam2):
    """현재 프레임 반환 (BGR)"""
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame