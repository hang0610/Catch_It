import math
import mujoco
# numpy provides import array and linear algebra utilities
import numpy as np

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

from spatialmath import base

from gymnasium import spaces

import quaternion

from omegaconf import DictConfig
from typing import Dict

class DynamicDelayBuffer:
    def __init__(self, maxlen):
        self.buffer = []
        self.maxlen = maxlen

    def append(self, item):
        if len(self.buffer) >= self.maxlen:
            self.buffer.pop(0)
        self.buffer.append(item)

    def set_maxlen(self, new_maxlen):
        self.maxlen = new_maxlen
        while len(self.buffer) > self.maxlen:
            self.buffer.pop(0)
    
    def clear(self):
        self.buffer.clear()
    
    def __getitem__(self, key):
        return self.buffer[key]

    def __repr__(self):
        return repr(self.buffer)

    def __len__(self):
        return self.maxlen

def omegaconf_to_dict(d: DictConfig)->Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret

def calculate_arm_Te(pose, quate):
    """
    Calculate the pose transform matrix of the end-effector.
    """
    if type(quate) is quaternion.quaternion:
        arm_ee_quat = quate
    else:
        arm_ee_quat = np.quaternion(quate[0], quate[1], quate[2], quate[3])
    # Calculate forward kinematics (Tep) for the target end-effector pose
    res = np.zeros(9)
    mujoco.mju_quat2Mat(res, np.array([arm_ee_quat.w, arm_ee_quat.x, arm_ee_quat.y, arm_ee_quat.z]))
    Te = np.eye(4)
    Te[:3,3] = pose
    Te[:3,:3] = res.reshape((3,3))
    return Te

def get_total_dimension(data):
    # print("type data: ", type(data))
    total_dimension = 0
    # If it is a dictionary, recursively process its values.
    if isinstance(data, spaces.Dict) or isinstance(data, dict):
        for value in data.values():
            total_dimension += get_total_dimension(value)
    # If it is a box, return the size of the box.
    elif isinstance(data, spaces.Box):
        return data.shape[0]
    # If it is an array, return the size of the array.
    elif isinstance(data, np.ndarray):
        return data.size
    # If it is a single element, return 1.
    else:
        return 1
    
    return total_dimension

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.
    """
    w, x, y, z = q
    return np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                     [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                     [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])

def cos_angle_between_vectors(v1, v2):
    """
    Calculate the cos angle between two vectors.
    """
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    # print("magnitude_v1: ", magnitude_v1)
    # If the velocity of the object is nearly zero, return 0
    if magnitude_v1 <= 0.3:
        return 0
    dot_product = np.dot(v1, v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    return cos_angle
    # angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    # angle_deg = np.degrees(angle_rad)
    # return angle_deg

def random_q(model: mujoco.MjModel, i: int = 1) -> np.ndarray:
    """
    Generate a random valid joint configuration

    :param i: number of configurations to generate

    Generates a random q vector within the joint limits of the model.
    """

    if i == 1:
        q = np.zeros(model.nv)

        for i in range(model.nv):
            q[i] = np.random.uniform(model.joint(i).range[0], model.joint(i).range[1])

    else:
        q = np.zeros((i, model.nv))

        for j in range(i):
            for i in range(model.nv):
                q[j, i] = np.random.uniform(model.joint(i).range[0], model.joint(i).range[1])

    return q

def angle_axis_python(T, Td):
    e = np.empty(6)
    e[:3] = Td[:3, -1] - T[:3, -1]
    R = Td[:3, :3] @ T[:3, :3].T
    li = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if base.iszerovec(li):
        # diagonal matrix case
        if np.trace(R) > 0:
            # (1,1,1) case
            a = np.zeros((3,))
        else:
            a = np.pi / 2 * (np.diag(R) + 1)
    else:
        # non-diagonal matrix case
        ln = base.norm(li)
        a = math.atan2(ln, np.trace(R) - 1) * li / ln

    e[3:] = a

    return e

def clip_norm(arr, upper_bound):
    """
    Clip the norm of the array.
    """
    norm = np.linalg.norm(arr)
    if norm > upper_bound:
        arr = arr / norm * upper_bound
    return arr

def relative_quaternion(q1, q2):
    # Calculate relative quaternion
    ## Note: DO NOT mess up the multiplication order of the quaternion :)
    quat_relative =  np.quaternion(q1[0], q1[1], q1[2], q1[3]).inverse() * np.quaternion(q2[0], q2[1], q2[2], q2[3])
    return np.array([quat_relative.w, quat_relative.x, quat_relative.y, quat_relative.z])

def relative_position(p1, p2, a):
    # Calculate relative position in global coordinates
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    # Rotate relative position around origin
    x = delta_x * np.cos(a) + delta_y * np.sin(a)
    y = - delta_x * np.sin(a) + delta_y * np.cos(a)

    return x, y

def quat2theta(qw,qz):
  """
  assume there is only rotation about the z-axis.
  """
  a = 2*np.arctan2(qz,qw)
  return np.arctan2( np.sin(a), np.cos(a) )