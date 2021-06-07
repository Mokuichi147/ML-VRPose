import time
import openvr


# SteamVR must be running
openvr.init(openvr.VRApplication_Background)
poses = []
while True:
    poses = openvr.VRSystem().getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseSeated, 0, poses)
    hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]

    temp = hmd_pose.mDeviceToAbsoluteTracking[0]
    print(f'{temp[0]:.3f}, {temp[1]:.3f}, {temp[2]:.3f}, {temp[3]:.3f}', end='\r', flush=True)
    time.sleep(0.0025)

openvr.shutdown()