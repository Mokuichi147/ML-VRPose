import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.camera import WebCam
from utils.ovr import VRTracker
from utils.pose import HumanPose


CAMERA_DEVICE = 0
MAX_POINTS = 100 * 2

# 3D graph
fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(5, -85)


object_points = []
image_points = []


with WebCam(CAMERA_DEVICE) as camera, HumanPose() as pose:
    # Camera Calibration
    camera.StartCalibration(10, 1.85, 7, 7, save_dir='calibration')
    # VR Tracker
    tracker = VRTracker()

    counter = 0

    while camera.IsOpened():
        if not camera.Read():
            print('Frame acquisition failed', end='\r', flush=True)
            continue
        elif not tracker.Read():
            print('Failed to get VR tracker', end='\r', flush=True)
            continue
        elif not pose.Read(camera):
            camera.Show('human pose')
            print('could not find anyone', end='\r', flush=True)
            continue
        else:
            counter = (counter + 1) % 1

        x = np.array([pose.landmarks.landmark[i].x for i in range(pose.landmark_count)])
        y = np.array([pose.landmarks.landmark[i].y for i in range(pose.landmark_count)]) * -1
        z = np.array([pose.landmarks.landmark[i].z for i in range(pose.landmark_count)])

        ax.cla()
        ax.scatter(x, z, y)
        plt.pause(0.001)

        pose.Draw()
        camera.Show('human pose')

        tracking_text  = '[HMD] X:{0.px:.3f}, Y:{0.py:.3f}, Z:{0.pz:.3f}\t'.format(tracker.hmd)
        tracking_text += '[L_CON] X:{0.px:.3f}, Y:{0.py:.3f}, Z:{0.pz:.3f}\t'.format(tracker.lcon)
        tracking_text += '[R_CON] X:{0.px:.3f}, Y:{0.py:.3f}, Z:{0.pz:.3f}'.format(tracker.rcon)
        print(tracking_text, end='\r', flush=True)


        if len(object_points) < MAX_POINTS and counter == 0 and pose.IsVisible(0.8, False):
            print('add')
            object_points.append(tracker.rcon.position)
            object_points.append(tracker.lcon.position)
            image_points.append(pose.rhand)
            image_points.append(pose.lhand)
            if len(object_points) == MAX_POINTS:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    np.array(object_points, dtype=np.float32),
                    np.array(image_points, dtype=np.float32),
                    camera.new_cam_mtx,
                    camera.calibration.dist_coeffs)
                print('rvec', rvec)
                print('tvec', tvec)
                print('inliers', inliers)
                
                imgpts, jac = cv2.projectPoints(np.array(object_points, dtype=np.float32), rvec, tvec, camera.new_cam_mtx, camera.calibration.dist_coeffs)
                
                sa = 0
                for i in range(len(object_points)):
                    for j in range(2):
                        sa += abs(imgpts[i][0][j] - image_points[i][j])
                    print(i, object_points[i], imgpts[i][0], image_points[i])
                sa /= MAX_POINTS
                print(sa)
                if sa > 50:
                    object_points = []
                    image_points = []
                else:
                    a = -np.matrix(cv2.Rodrigues(rvec)[0]).T * np.matrix(tvec)
                    print('Position', a)
                    break

        if camera.Wait():
            break