import numpy as np
import openvr


def GetPosition(matrix44):
    x = matrix44[0,3]
    y = matrix44[1,3]
    z = matrix44[2,3]
    return np.array([x, y, z])

def GetRotaion(matrix44):
    qw = np.sqrt(max(0, 1 + matrix44[0,0] + matrix44[1,1] + matrix44[2,2])) / 2
    qx = np.sqrt(max(0, 1 + matrix44[0,0] - matrix44[1,1] - matrix44[2,2])) / 2
    qy = np.sqrt(max(0, 1 - matrix44[0,0] + matrix44[1,1] - matrix44[2,2])) / 2
    qz = np.sqrt(max(0, 1 - matrix44[0,0] - matrix44[1,1] + matrix44[2,2])) / 2
    qx = abs(qx) if matrix44[2,1] - matrix44[1,2] >= 0 else -abs(qx)
    qy = abs(qy) if matrix44[0,2] - matrix44[2,0] >= 0 else -abs(qy)
    qz = abs(qz) if matrix44[1,0] - matrix44[0,1] >= 0 else -abs(qz)
    return np.array([qx, qy, qz, qw])

def GetScale(matrix44):
    x = np.sqrt(matrix44[0,0]**2 + matrix44[0,1]**2 + matrix44[0,2]**2)
    y = np.sqrt(matrix44[1,0]**2 + matrix44[1,1]**2 + matrix44[1,2]**2)
    z = np.sqrt(matrix44[2,0]**2 + matrix44[2,1]**2 + matrix44[2,2]**2)
    return np.array([x, y, z])


class Transform:
    def __init__(self, pose_34):
        self.matrix = np.eye(4)

        self.matrix[0, 0] =  pose_34[0][0]
        self.matrix[0, 1] =  pose_34[0][1]
        self.matrix[0, 2] = -pose_34[0][2]
        self.matrix[0, 3] =  pose_34[0][3]

        self.matrix[1, 0] =  pose_34[1][0]
        self.matrix[1, 1] =  pose_34[1][1]
        self.matrix[1, 2] = -pose_34[1][2]
        self.matrix[1, 3] =  pose_34[1][3]

        self.matrix[2, 0] = -pose_34[2][0]
        self.matrix[2, 1] = -pose_34[2][1]
        self.matrix[2, 2] =  pose_34[2][2]
        self.matrix[2, 3] = -pose_34[2][3]

        self.SetPosition(self.matrix)
        self.SetRotation(self.matrix)
        self.SetScale(self.matrix)
    
    def SetPosition(self, matrix44):
        self.position = GetPosition(matrix44)
        self.px = self.position[0]
        self.py = self.position[1]
        self.pz = self.position[2]

    def SetRotation(self, matrix44):
        self.rotation = GetRotaion(matrix44)
        self.qx = self.rotation[0]
        self.qy = self.rotation[1]
        self.qz = self.rotation[2]
        self.qw = self.rotation[3]

    def SetScale(self, matrix44):
        self.scale = GetScale(matrix44)
        self.sx = self.scale[0]
        self.sy = self.scale[1]
        self.sz = self.scale[2]


class Controller:
    def __init__(self, openvr_system):
        self.openvr_system = openvr_system
        self.right = -1
        self.left  = -1
        self.Update()

    def Update(self):
        for device_index in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self.openvr_system.getTrackedDeviceClass(device_index)
            if device_class != openvr.TrackedDeviceClass_Controller:
                continue

            device_role = self.openvr_system.getControllerRoleForTrackedDeviceIndex(device_index)
            if device_role == openvr.TrackedControllerRole_RightHand:
                self.right = device_index
            elif device_role == openvr.TrackedControllerRole_RightHand:
                self.left  = device_index


class Tracking:
    def __init__(self):
        openvr.init(openvr.VRApplication_Background)
        self.openvr_system = openvr.VRSystem()
        self.controller = Controller(self.openvr_system)
        self.poses = []
        
        self.hmd  = Transform(np.zeros([3,4]))
        self.rcon = Transform(np.zeros([3,4]))
        self.lcon = Transform(np.zeros([3,4]))
    
    def Update(self):
        self.poses = self.openvr_system.getDeviceToAbsoluteTrackingPose(
                                              openvr.TrackingUniverseSeated, 0, self.poses)

        _hmd_pose = self.poses[self.openvr_system.k_unTrackedDeviceIndex_Hmd]
        _controller_right_pose = self.poses[self.controller.right]
        _controller_left_pose  = self.poses[self.controller.left]

        self.hmd  = Transform(_hmd_pose)
        self.rcon = Transform(_controller_right_pose)
        self.lcon = Transform(_controller_left_pose)