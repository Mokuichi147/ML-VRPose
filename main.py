import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mpl_toolkits.mplot3d import Axes3D


CAMERA_DEVICE = 0
LANDMARK_COUNT = 33

# MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 3D graph
fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(5, -85)

# Camera Calibration
capture_device = cv2.VideoCapture(CAMERA_DEVICE)
mtx  = np.load('calibration/mtx.npy')
dist = np.load('calibration/dist.npy')
success, frame = capture_device.read()
image_size = (frame.shape[1], frame.shape[0])
new_mtx, (cam_x,cam_y,cam_w,cam_h) = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, 0, image_size)
map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, image_size, cv2.CV_32FC1)


with mp_pose.Pose(
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5) as pose:
    while capture_device.isOpened():
        success, frame = capture_device.read()
        if not success:
            print('Frame acquisition failed')
            continue

        calibration_frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w]

        image = cv2.cvtColor(cv2.flip(calibration_frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        if results.pose_landmarks == None:
            cv2.imshow('human pose', image)
            print('could not find anyone')
            continue

        x = np.array([results.pose_landmarks.landmark[i].x for i in range(LANDMARK_COUNT)])
        y = np.array([results.pose_landmarks.landmark[i].y for i in range(LANDMARK_COUNT)]) * -1
        z = np.array([results.pose_landmarks.landmark[i].z for i in range(LANDMARK_COUNT)])

        ax.cla()
        ax.scatter(x, z, y)
        plt.pause(0.001)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('human pose', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture_device.release()
cv2.destroyAllWindows()