import cv2
import numpy as np
from os import makedirs


class WebCam:
    def __init__(self, camera_device=0):
        self.camera_device = camera_device
        self.capture = cv2.VideoCapture(self.camera_device)
        
        self.success = False
        self.frame = None
        
        self.w   = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h   = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

    def __del__(self):
        self.camera_device.release()
        cv2.destroyAllWindows()

    def IsOpened(self):
        return self.camera_device.isOpened()

    def Read(self):
        self.success, self.frame = self.capture.read()
        return self.success

    def Show(self, window_title):
        cv2.imshow(window_title, self.frame)

    def Wait(self, ms=1, key_code='q'):
        return cv2.waitKey(ms) & 0xFF == ord(key_code)