import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mpl_toolkits.mplot3d import Axes3D
from utils.camera import WebCam


CAMERA_DEVICE = 0
LANDMARK_COUNT = 33

# MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 3D graph
fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(5, -85)

camera = WebCam(CAMERA_DEVICE)
# Camera Calibration
camera.StartCalibration(10, 1.85, 7, 7, save_dir='calibration')


with mp_pose.Pose(
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5) as pose:
    while camera.IsOpened():
        if not camera.Read():
            print('Frame acquisition failed', end='\r', flush=True)
            continue
        
        camera.FlipFrame()
        camera.ConvertRGB()
        camera.FrameWriteable(False)
        # Pose estimation
        results = pose.process(camera.frame)
        camera.FrameWriteable(True)
        camera.ConvertBGR()

        if results.pose_landmarks == None:
            camera.Show('human pose')
            print('could not find anyone', end='\r', flush=True)
            continue

        x = np.array([results.pose_landmarks.landmark[i].x for i in range(LANDMARK_COUNT)])
        y = np.array([results.pose_landmarks.landmark[i].y for i in range(LANDMARK_COUNT)]) * -1
        z = np.array([results.pose_landmarks.landmark[i].z for i in range(LANDMARK_COUNT)])

        ax.cla()
        ax.scatter(x, z, y)
        plt.pause(0.001)

        mp_drawing.draw_landmarks(camera.frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        camera.Show('human pose')

        if camera.Wait():
            break