import cv2
import numpy as np
from os import makedirs


class Calibration:
    def __init__(self, frame_count, box_size, row_cross, column_cross, root_dir='calibration'):
        self.frame_count = frame_count
        self.box_size = box_size
        self.r_cross = row_cross
        self.c_cross = column_cross
        self.root_dir = root_dir

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.object_point = np.zeros((self.r_cross * self.c_cross, 3), np.float32)
        self.object_point[:,:2] = np.mgrid[0:self.c_cross, 0:self.r_cross].T.reshape(-1, 2)
        self.object_point *= self.box_size
        self.object_points = []
        self.image_points  = []

        self.cam_mtx = None
        self.dist_coeffs = None

    def IsActivated(self):
        if self.cam_mtx == None or self.dist_coeffs == None:
            return False
        return True

    def IsCaptureCompleted(self):
        return len(self.image_points) > self.frame_count or self.IsActivated()

    def AddCapture(self, bgr_frame):
        if self.IsCaptureCompleted():
            return False
        _gray = cv2.cvtColorA(bgr_frame, cv2.COLOR_BGR2GRAY)
        _success, _corners = cv2.findChessboardCorners(_gray, (self.c_cross, self.r_cross), None)
        if not _success:
            return False
        self.object_points.append(self.object_point)
        self.image_points.append(_corners)
        if self.IsCompleted():
            self.Save()
        return True

    def Load(self):
        self.cam_mtx = np.load(f'{self.root_dir}/cam_mtx.npy')
        self.dist_coeffs = np.load(f'{self.root_dir}/dist_coeffs.npy')

    def Save(self):
        _success, _cam_mtx, _dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            (self.h, self.w),
            None,
            criteria=self.criteria)
        
        if not _success:
            print('Calibration failed!')
            return False

        self.cam_mtx = _cam_mtx
        self.dist_coeffs = _dist_coeffs.ravel()

        makedirs(self.root_dir, exist_ok=True)
        np.save(f'{self.root_dir}/cam_mtx', _cam_mtx)
        np.save(f'{self.root_dir}/dist_coeffs', _dist_coeffs.ravel())
        return True


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