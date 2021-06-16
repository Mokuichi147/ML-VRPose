import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.camera import WebCam
from utils.pose import HumanPose


CAMERA_DEVICE = 0
LANDMARK_COUNT = 33

# 3D graph
fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(5, -85)

with WebCam(CAMERA_DEVICE) as camera, HumanPose(landmark_count=LANDMARK_COUNT) as pose:
    # Camera Calibration
    camera.StartCalibration(10, 1.85, 7, 7, save_dir='calibration')

    while camera.IsOpened():
        if not camera.Read():
            print('Frame acquisition failed', end='\r', flush=True)
            continue
        elif not pose.Read(camera):
            camera.Show('human pose')
            print('could not find anyone', end='\r', flush=True)
            continue

        x = np.array([pose.landmarks.landmark[i].x for i in range(LANDMARK_COUNT)])
        y = np.array([pose.landmarks.landmark[i].y for i in range(LANDMARK_COUNT)]) * -1
        z = np.array([pose.landmarks.landmark[i].z for i in range(LANDMARK_COUNT)])

        ax.cla()
        ax.scatter(x, z, y)
        plt.pause(0.001)

        pose.Draw()
        camera.Show('human pose')

        if camera.Wait():
            break