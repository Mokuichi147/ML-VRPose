from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose


class HumanPose:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, landmark_count=33):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.landmark_count = landmark_count

    def __enter__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence = self.min_detection_confidence,
            min_tracking_confidence  = self.min_tracking_confidence)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

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

        return self.landmarks != None

    def Draw(self):
        mp_drawing.draw_landmarks(self.frame, self.landmarks, mp_pose.POSE_CONNECTIONS)