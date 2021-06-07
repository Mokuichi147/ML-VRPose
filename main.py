import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mpl_toolkits.mplot3d import Axes3D


CAMERA_DEVICE = 2
LANDMARK_COUNT = 33

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(5, -85)

with mp_pose.Pose(
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5) as pose:
    capture_device = cv2.VideoCapture(CAMERA_DEVICE)
    while capture_device.isOpened():
        success, frame = capture_device.read()
        if not success:
            print('Frame acquisition failed')
            continue

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
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