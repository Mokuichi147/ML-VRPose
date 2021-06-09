import time
import openvr

from utils import ovr_utils


# SteamVR must be running
openvr.init(openvr.VRApplication_Background)
poses = []
while True:
    if not openvr.isHmdPresent():
        continue

    poses = openvr.VRSystem().getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseSeated, 0, poses)
    hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
    controller = ovr_utils.Controller()
    controller_right_pose = poses[controller.right]
    controller_left_pose = poses[controller.left]

    hmd_transform = ovr_utils.Transform(hmd_pose.mDeviceToAbsoluteTracking)
    print('[Position] X:{0.px:.3f}, Y:{0.py:.3f}, Z:{0.pz:.3f}'.format(hmd_transform), end='\r', flush=True)
    time.sleep(0.0025)

openvr.shutdown()