import time
import openvr


openvr.init(openvr.VRApplication_Scene)
poses = []
while True:
    poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
    hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]

    temp = hmd_pose.mDeviceToAbsoluteTracking[0]
    print(f'{temp[0]:.3f}, {temp[1]:.3f}, {temp[2]:.3f}, {temp[3]:.3f}', end='\r', flush=True)
    time.sleep(0.0025)

openvr.shutdown()
