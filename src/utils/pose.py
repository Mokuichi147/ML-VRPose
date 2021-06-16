import numpy as np
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose


class HumanPose:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, landmark_count=33):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.landmark_count = landmark_count

        self.landmarks = None
        self.frame = None

        self.head = np.zeros(3)
        self.lhand = np.zeros(3)
        self.rhand = np.zeros(3)

    def __enter__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence = self.min_detection_confidence,
            min_tracking_confidence  = self.min_tracking_confidence)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pose.close()

    def GetPositionArray(self, index):
        _position = self.landmarks.landmark[index]
        return np.array([_position.x, _position.y, _position.z])

    def Read(self, camera, flip=True):
        if flip:
            camera.FlipFrame()
        camera.ConvertRGB()
        camera.FrameWriteable(False)
        # Pose estimation
        self.landmarks = self.pose.process(camera.frame).pose_landmarks
        camera.FrameWriteable(True)
        camera.ConvertBGR()
        self.frame = camera.frame

        if self.landmarks == None:
            return False

        self.head  = self.GetPositionArray(mp_pose.PoseLandmark.NOSE)
        _left  = mp_pose.PoseLandmark.LEFT_WRIST if not flip else mp_pose.PoseLandmark.RIGHT_WRIST
        _right = mp_pose.PoseLandmark.RIGHT_WRIST if not flip else mp_pose.PoseLandmark.LEFT_WRIST
        self.lhand = self.GetPositionArray(_left)
        self.rhand = self.GetPositionArray(_right)
        return True

    def Draw(self):
        if self.landmarks != None:
            mp_drawing.draw_landmarks(self.frame, self.landmarks, mp_pose.POSE_CONNECTIONS)