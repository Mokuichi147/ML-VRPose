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

        self.head_index = mp_pose.PoseLandmark.NOSE
        self.lhand_index = mp_pose.PoseLandmark.LEFT_WRIST
        self.rhand_index = mp_pose.PoseLandmark.RIGHT_WRIST

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

    def GetImagePosition(self, index):
        _position = self.landmarks.landmark[index]
        _px = round(self.frame.shape[1] * _position.x)
        _py = round(self.frame.shape[0] * _position.y)
        return np.array([_px, _py])

    def IsVisible(self, threshold, full_body=True):
        if full_body:
            _check_list = [i for i in range(self.landmark_count)]
        else:
            _check_list = [self.head_index, self.lhand_index, self.rhand_index]

        for i in _check_list:
            if self.landmarks.landmark[i].visibility < threshold:
                return False
        return True

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

        self.head  = self.GetImagePosition(self.head_index)
        self.lhand_index  = mp_pose.PoseLandmark.LEFT_WRIST if not flip else mp_pose.PoseLandmark.RIGHT_WRIST
        self.rhand_index = mp_pose.PoseLandmark.RIGHT_WRIST if not flip else mp_pose.PoseLandmark.LEFT_WRIST
        self.lhand = self.GetImagePosition(self.lhand_index)
        self.rhand = self.GetImagePosition(self.rhand_index)
        return True

    def Draw(self):
        if self.landmarks != None:
            mp_drawing.draw_landmarks(self.frame, self.landmarks, mp_pose.POSE_CONNECTIONS)