import cv2
import numpy as np
from os import makedirs
from os.path import isfile


class Calibration:
    def __init__(self, frame_count, box_size_cm, row_cross, column_cross, save_dir='calibration'):
        self.frame_count = frame_count
        self.box_size = box_size_cm
        self.r_cross = row_cross
        self.c_cross = column_cross
        self.save_dir = save_dir

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

    def ShowProgress(self):
        print(f'[progress] {len(self.image_points)} out of {self.frame_count}', end='\r', flush=True)

    def Load(self):
        if isfile(f'{self.save_dir}/cam_mtx.npy'):
            self.cam_mtx = np.load(f'{self.save_dir}/cam_mtx.npy')
        if isfile(f'{self.save_dir}/dist_coeffs.npy'):
            self.dist_coeffs = np.load(f'{self.save_dir}/dist_coeffs.npy')

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

        makedirs(self.save_dir, exist_ok=True)
        np.save(f'{self.save_dir}/cam_mtx', _cam_mtx)
        np.save(f'{self.save_dir}/dist_coeffs', _dist_coeffs.ravel())
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

        self.is_calibration = False
        self.new_cam_mtx = None
        self.cx = None
        self.cy = None
        self.cw = None
        self.ch = None
        self.map_x = None
        self.map_y = None

    def __del__(self):
        self.camera_device.release()
        cv2.destroyAllWindows()

    def IsOpened(self):
        return self.camera_device.isOpened()

    def Read(self):
        self.success, self.frame = self.capture.read()
        if self.success and self.is_calibration:
            self.frame = cv2.remap(
                self.frame,
                self.map_x,
                self.map_y,
                cv2.INTER_LINEAR)[self.cy:self.cy+self.ch, self.cx:self.cx+self.cw]
        return self.success

    def FlipFrame(self, mode=1):
        self.frame = cv2.flip(self.frame, mode)

    def FrameWriteable(self, bool):
        self.frame.flags.writeable = bool

    def ConvertRGB(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def ConvertBGR(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

    def Show(self, window_title):
        cv2.imshow(window_title, self.frame)

    def Wait(self, ms=1, key_code='q'):
        return cv2.waitKey(ms) & 0xFF == ord(key_code)

    def StartCalibration(self, frame_count, box_size_cm, row_cross, column_cross, save_dir='calibration'):
        self.calibration = Calibration(frame_count, box_size_cm, row_cross, column_cross, save_dir=save_dir)
        self.calibration.Load()
        if not self.calibration.IsActivated():
            self.CalibrationMode()

        self.new_cam_mtx, (self.cx,self.cy,self.cw,self.ch) = cv2.getOptimalNewCameraMatrix(
            self.calibration.cam_mtx,
            self.calibration.dist_coeffs,
            (self.w, self.h),
            0,
            (self.w, self.h))

        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.calibration.cam_mtx,
            self.calibration.dist_coeffs,
            None,
            self.new_cam_mtx,
            (self.w, self.h),
            cv2.CV_32FC1)

        self.is_calibration = True

    def CalibrationMode(self):
        print('Start calibration mode')
        print('Press the spacebar')
        while self.IsOpened() and self.calibration.IsCaptureCompleted():
            self.calibration.ShowProgress()

            self.Read()
            self.Show()

            if self.Wait(key_code=' '):
                self.calibration.AddCapture(self.frame)