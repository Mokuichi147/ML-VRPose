import cv2
import numpy as np
import os


CAMERA_DEVICE = 0
FRAME_COUNT = 10
ROW_BOX_COUNT = 8
COLUMN_BOX_COUNT = 8
BOX_SIZE_CM = 1.85

r_cross = ROW_BOX_COUNT - 1
c_cross = COLUMN_BOX_COUNT - 1

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
object_point = np.zeros((r_cross*c_cross,3), np.float32)
object_point[:,:2] = np.mgrid[0:c_cross,0:r_cross].T.reshape(-1,2)
object_point *= BOX_SIZE_CM
object_points = []
image_points  = []

capture_device = cv2.VideoCapture(CAMERA_DEVICE)
while capture_device.isOpened() and len(image_points) < FRAME_COUNT:
    print(f'[progress] {len(image_points)} out of {FRAME_COUNT}', end='\r', flush=True)
    success, frame = capture_device.read()
    if not success:
        print('Frame acquisition failed')
        continue

    cv2.imshow('camera calibration', frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(gray, (c_cross,r_cross), None)
        if not success:
            continue
        object_points.append(object_point)
        corners2 = cv2.cornerSubPix(gray,corners, (5,5), (-1,-1), criteria)
        image_points.append(corners)

if len(image_points) == FRAME_COUNT:
    success, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    os.makedirs('calibration', exist_ok=True)
    np.save('calibration/mtx', mtx)
    np.save('calibration/dist', dist.ravel())
    print('Done!')

capture_device.release()
cv2.destroyAllWindows()