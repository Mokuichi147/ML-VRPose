import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.camera import WebCam
from utils.pose import HumanPose


CAMERA_DEVICE = 0

# 3D graph
fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(5, -85)

with WebCam(CAMERA_DEVICE) as camera, HumanPose() as pose:
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

        x = np.array([pose.landmarks.landmark[i].x for i in range(pose.landmark_count)])
        y = np.array([pose.landmarks.landmark[i].y for i in range(pose.landmark_count)]) * -1
        z = np.array([pose.landmarks.landmark[i].z for i in range(pose.landmark_count)])

        ax.cla()
        ax.scatter(x, z, y)
        plt.pause(0.001)

        pose.Draw()
        camera.Show('human pose')

        tracking_text  = '[Head] X:{0[0]:.3f}, Y:{0[1]:.3f}\t'.format(pose.head)
        tracking_text += '[L_Hand] X:{0[0]:.3f}, Y:{0[1]:.3f}\t'.format(pose.lhand)
        tracking_text += '[R_Hand] X:{0[0]:.3f}, Y:{0[1]:.3f}'.format(pose.rhand)
        print(tracking_text, end='\r', flush=True)

        if camera.Wait():
            break