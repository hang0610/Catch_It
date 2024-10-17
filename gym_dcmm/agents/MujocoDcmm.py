"""
Author: Yuanhang Zhang
Version@2024-10-17
All Rights Reserved
ABOUT: this file constains the basic class of the DexCatch with Mobile Manipulation (DCMM) in the MuJoCo simulation environment.
"""
import os, sys
sys.path.append(os.path.abspath('../'))
import copy
import configs.env.DcmmCfg as DcmmCfg
import mujoco
from utils.util import calculate_arm_Te
from utils.pid import PID
import numpy as np
from utils.ik_pkg.ik_arm import IKArm
from utils.ik_pkg.ik_base import IKBase
from scipy.spatial.transform import Rotation as R
from collections import deque
import xml.etree.ElementTree as ET

# Function to convert XML file to string
def xml_to_string(file_path):
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Convert the XML element tree to a string
        xml_str = ET.tostring(root, encoding='unicode')
        
        return xml_str
    except Exception as e:
        print(f"Error: {e}")
        return None

DEBUG_ARM = False
DEBUG_BASE = False


class MJ_DCMM(object):
    """
    Class of the DexCatch with Mobile Manipulation (DCMM)
    in the MuJoCo simulation environment.

    Args:
    - model: the MuJoCo model of the Dcmm
    - model_arm: the MuJoCo model of the arm
    - viewer: whether to show the viewer of the simulation
    - object_name: the name of the object in the MuJoCo model
    - timestep: the simulation timestep
    - open_viewer: whether to open the viewer initially

    """
    def __init__(self, 
                 model=None, 
                 model_arm=None, 
                 viewer=True, 
                 object_name='object',
                 object_eval=False, 
                 timestep=0.002):
        self.viewer = None
        self.open_viewer = viewer
        # Load the MuJoCo model
        if model is None:
            if not object_eval: model_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_DCMM_LEAP_OBJECT_PATH)
            else: model_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_DCMM_LEAP_UNSEEN_OBJECT_PATH)
            self.model_xml_string = xml_to_string(model_path)
        else:
            self.model = model
        if model_arm is None:
            model_arm_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_ARM_PATH)
            self.model_arm = mujoco.MjModel.from_xml_path(model_arm_path)
        else:
            self.model_arm = model_arm
        self.model = mujoco.MjModel.from_xml_string(self.model_xml_string)
        self.model.opt.timestep = timestep
        self.model_arm.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.data_arm = mujoco.MjData(self.model_arm)
        self.data.qpos[15:21] = DcmmCfg.arm_joints[:]
        self.data.qpos[21:37] = DcmmCfg.hand_joints[:]
        self.data_arm.qpos[0:6] = DcmmCfg.arm_joints[:]

        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_forward(self.model_arm, self.data_arm)
        self.arm_base_pos = self.data.body("arm_base").xpos
        self.current_ee_pos = copy.deepcopy(self.data_arm.body("link6").xpos)
        self.current_ee_quat = copy.deepcopy(self.data_arm.body("link6").xquat)

        ## Get the joint ID for the body, base, arm, hand and object
        # Note: The joint id of the mm body is 0 by default
        try:
            _ = self.data.body(object_name)
        except:
            print("The object name is not found in the model!\
                  \nPlease check the object name in the .xml file.")
            raise ValueError
        self.object_name = object_name
        # Get the geom id of the hand, the floor and the object
        self.hand_start_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'mcp_joint') - 1
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.object_name)

        # Mobile Base Control
        self.rp_base = np.zeros(3)
        self.rp_ref_base = np.zeros(3)
        self.drive_pid = PID("drive", DcmmCfg.Kp_drive, DcmmCfg.Ki_drive, DcmmCfg.Kd_drive, dim=4, llim=DcmmCfg.llim_drive, ulim=DcmmCfg.ulim_drive, debug=False)
        self.steer_pid = PID("steer", DcmmCfg.Kp_steer, DcmmCfg.Ki_steer, DcmmCfg.Kd_steer, dim=4, llim=DcmmCfg.llim_steer, ulim=DcmmCfg.ulim_steer, debug=False)
        self.arm_pid = PID("arm", DcmmCfg.Kp_arm, DcmmCfg.Ki_arm, DcmmCfg.Kd_arm, dim=6, llim=DcmmCfg.llim_arm, ulim=DcmmCfg.ulim_arm, debug=False)
        self.hand_pid = PID("hand", DcmmCfg.Kp_hand, DcmmCfg.Ki_hand, DcmmCfg.Kd_hand, dim=16, llim=DcmmCfg.llim_hand, ulim=DcmmCfg.ulim_hand, debug=False)
        self.cmd_lin_y = 0.0
        self.cmd_lin_x = 0.0
        self.arm_act = False
        self.steer_ang = np.array([0.0, 0.0, 0.0, 0.0])
        self.drive_vel = np.array([0.0, 0.0, 0.0, 0.0])

        ## Define Inverse Kinematics Solver for the Arm
        self.ik_arm = IKArm(solver_type=DcmmCfg.ik_config["solver_type"], ilimit=DcmmCfg.ik_config["ilimit"], 
                            ps=DcmmCfg.ik_config["ps"], λΣ=DcmmCfg.ik_config["λΣ"], tol=DcmmCfg.ik_config["ee_tol"])

        ## Initialize the camera parameters
        self.model.vis.global_.offwidth = DcmmCfg.cam_config["width"]
        self.model.vis.global_.offheight = DcmmCfg.cam_config["height"]
        self.create_camera_data(DcmmCfg.cam_config["width"], DcmmCfg.cam_config["height"], DcmmCfg.cam_config["name"])

        ## Initialize the target velocity of the mobile base
        self.target_base_vel = np.zeros(3)
        self.target_arm_qpos = np.zeros(6)
        self.target_hand_qpos = np.zeros(16)
        ## Initialize the target joint positions of the arm
        self.target_arm_qpos[:] = DcmmCfg.arm_joints[:]
        ## Initialize the target joint positions of the hand
        self.target_hand_qpos[:] = DcmmCfg.hand_joints[:]

        self.ik_solution = np.zeros(6)

        self.vel_history = deque(maxlen=4)  # store the last 2 velocities
        self.vel_init = False

        self.drive_ctrlrange = self.model.actuator(4).ctrlrange
        self.steer_ctrlrange = self.model.actuator(0).ctrlrange

    def show_model_info(self):
        """
        Displays relevant model info for the user, namely bodies, joints, actuators, as well as their IDs and ranges.
        Also gives info on which actuators control which joints and which joints are included in the kinematic chain,
        as well as the PID controller info for each actuator.
        """

        print("\nNumber of bodies: {}".format(self.model.nbody))
        for i in range(self.model.nbody):
            print("Body ID: {}, Body Name: {}".format(i, self.model.body(i).name))

        print("\nNumber of joints: {}".format(self.model.njnt))
        for i in range(self.model.njnt):
            print(
                "Joint ID: {}, Joint Name: {}, Limits: {}, Damping: {}".format(
                    i, self.model.joint(i).name, self.model.jnt_range[i], self.model.dof_damping[i]
                )
            )

        print("\nNumber of Actuators: {}".format(len(self.data.ctrl)))
        for i in range(len(self.data.ctrl)):
            print(
                "Actuator ID: {}, Actuator Name: {}, Controlled Joint: {}, Control Range: {}".format(
                    i,
                    self.model.actuator(i).name,
                    self.model.joint(self.model.actuator(i).trnid[0]).name,
                    self.model.actuator(i).ctrlrange,
                )
            )
        print("\nMobile Base PID Info: \n")
        print(
            "Drive, P: {}, I: {}, D: {}".format(
                self.drive_pid.Kp,
                self.drive_pid.Ki,
                self.drive_pid.Kd,
            )
        )
        print(
            "Steer, P: {}, I: {}, D: {}".format(
                self.steer_pid.Kp,
                self.steer_pid.Ki,
                self.steer_pid.Kd,
            )
        )
        print("\nArm PID Info: \n")
        print(
            "P: {}, I: {}, D: {}".format(
                self.arm_pid.Kp,
                self.arm_pid.Ki,
                self.arm_pid.Kd,
            )
        )
        print("\nHand PID Info: \n")
        print(
            "P: {}, I: {}, D: {}".format(
                self.hand_pid.Kp,
                self.hand_pid.Ki,
                self.hand_pid.Kd,
            )
        )

        print("\nCamera Info: \n")
        for i in range(self.model.ncam):
            print(
                "Camera ID: {}, Camera Name: {}, Camera Mode: {}, Camera FOV (y, degrees): {}, Position: {}, Orientation: {}, \n Intrinsic Matrix: \n{}".format(
                    i,
                    self.model.camera(i).name,
                    self.model.cam_mode[i],
                    self.model.cam_fovy[i],
                    self.model.cam_pos[i],
                    self.model.cam_quat[i],
                    # self.model.cam_pos0[i],
                    # self.model.cam_mat0[i].reshape((3, 3)),
                    self.cam_matrix,
                )
            )
        print("\nSimulation Timestep: ", self.model.opt.timestep)
    
    def move_base_vel(self, target_base_vel):
        self.steer_ang, self.drive_vel = IKBase(target_base_vel[0], target_base_vel[1], target_base_vel[2])
        ####################
        ## No bugs so far ##
        ####################
        # Mobile base steering and driving control 
        # TODO: angular velocity is not correct when the robot is self-rotating.
        current_steer_pos = np.array([self.data.joint("steer_fl").qpos[0],
                                      self.data.joint("steer_fr").qpos[0], 
                                      self.data.joint("steer_rl").qpos[0],
                                      self.data.joint("steer_rr").qpos[0]])
        current_drive_vel = np.array([self.data.joint("drive_fl").qvel[0],
                                      self.data.joint("drive_fr").qvel[0], 
                                      self.data.joint("drive_rl").qvel[0],
                                      self.data.joint("drive_rr").qvel[0]])
        mv_steer = self.steer_pid.update(self.steer_ang, current_steer_pos, self.data.time)
        mv_drive = self.drive_pid.update(self.drive_vel, current_drive_vel, self.data.time)
        if np.all(current_drive_vel > 0.0) and np.all(current_drive_vel < self.drive_vel):
            mv_drive = np.clip(mv_drive, 0, self.drive_ctrlrange[1] / 10.0)
        if np.all(current_drive_vel < 0.0) and np.all(current_drive_vel > self.drive_vel):
            mv_drive = np.clip(mv_drive, self.drive_ctrlrange[0] / 10.0, 0)
        
        mv_steer = np.clip(mv_steer, self.steer_ctrlrange[0], self.steer_ctrlrange[1])
        
        return mv_steer, mv_drive
    
    def move_ee_pose(self, delta_pose):
        """
        Move the end-effector to the target pose.
        delta_pose[0:3]: delta x,y,z
        delta_pose[3:6]: delta euler angles roll, pitch, yaw

        Return:
        - The target joint positions of the arm
        """
        self.current_ee_pos[:] = self.data_arm.body("link6").xpos[:]
        self.current_ee_quat[:] = self.data_arm.body("link6").xquat[:]
        target_pos = self.current_ee_pos + delta_pose[0:3]
        r_delta = R.from_euler('zxy', delta_pose[3:6])
        r_current = R.from_quat(self.current_ee_quat)
        target_quat = (r_delta * r_current).as_quat()
        result_QP = self.ik_arm_solve(target_pos, target_quat)
        if DEBUG_ARM: print("result_QP: ", result_QP)
        # Update the qpos of the arm with the IK solution
        self.data_arm.qpos[0:6] = result_QP[0]
        mujoco.mj_fwdPosition(self.model_arm, self.data_arm)
        
        # Compute the ee_length
        relative_ee_pos = target_pos - self.data_arm.body("arm_base").xpos
        ee_length = np.linalg.norm(relative_ee_pos)

        return result_QP, ee_length
    
    def ik_arm_solve(self, target_pose, target_quate):
        """
        Solve the IK problem for the arm.
        """
        # Update the arm joint position to the previous one
        Tep = calculate_arm_Te(target_pose, target_quate)
        if DEBUG_ARM: print("Tep: ", Tep)
        result_QP = self.ik_arm.solve(self.model_arm, self.data_arm, Tep, self.data_arm.qpos[0:6])
        return result_QP

    def set_throw_pos_vel(self, 
                          pose = np.array([0, 0, 0, 1, 0, 0, 0]), 
                          velocity = np.array([0, 0, 0, 0, 0, 0])):
        self.data.qpos[37:44] = pose
        self.data.qvel[36:42] = velocity

    def action_hand2qpos(self, action_hand):
        """
        Convert the action of the hand to the joint positions.
        """
        # Thumb
        self.target_hand_qpos[13] += action_hand[9]
        self.target_hand_qpos[14] += action_hand[10]
        self.target_hand_qpos[15] += action_hand[11]
        # Other Three Fingers
        self.target_hand_qpos[0] += action_hand[0]
        self.target_hand_qpos[2] += action_hand[1]
        self.target_hand_qpos[3] += action_hand[2]
        self.target_hand_qpos[4] += action_hand[3]
        self.target_hand_qpos[6] += action_hand[4]
        self.target_hand_qpos[7] += action_hand[5]
        self.target_hand_qpos[8] += action_hand[6]
        self.target_hand_qpos[10] += action_hand[7]
        self.target_hand_qpos[11] += action_hand[8]

    def pixel_2_world(self, pixel_x, pixel_y, depth, camera="top"):
        """
        Converts pixel coordinates into world coordinates.

        Args:
            pixel_x: X-coordinate in pixel space.
            pixel_y: Y-coordinate in pixel space.
            depth: Depth value corresponding to the pixel.
            camera: Name of camera used to obtain the image.
        """

        if not self.cam_init:
            self.create_camera_data(DcmmCfg.cam_config["width"], DcmmCfg.cam_config["height"], camera)

        # Create coordinate vector
        pixel_coord = np.array([pixel_x, 
                                pixel_y, 
                                1]) * (depth)
        
        # Get position relative to camera
        pos_c = np.linalg.inv(self.cam_matrix) @ pixel_coord
        # Transform to the global frame axis
        pos_c[1] *= -1
        pos_c[1], pos_c[2] = pos_c[2], pos_c[1]
        # Get world position
        pos_w = self.cam_rot_mat @ (pos_c) + self.cam_pos

        return pos_c, pos_w

    def depth_2_meters(self, depth):
        """
        Converts the depth array delivered by MuJoCo (values between 0 and 1) into actual m values.

        Args:
            depth: The depth array to be converted.
        """

        extend = self.model.stat.extent
        near = self.model.vis.map.znear * extend
        far = self.model.vis.map.zfar * extend

        return near / (1 - depth * (1 - near / far))

    def create_camera_data(self, width, height, camera):
        """
        Initializes all camera parameters that only need to be calculated once.
        """

        cam_id = self.model.camera(camera).id
        # Get field of view
        fovy = self.model.cam_fovy[cam_id]
        # Calculate focal length
        f = 0.5 * height / np.tan(fovy * np.pi / 360)
        # Construct camera matrix
        self.cam_matrix = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        # Rotation of camera in world coordinates
        self.cam_rot_mat = self.model.cam_mat0[cam_id]
        self.cam_rot_mat = np.reshape(self.cam_rot_mat, (3, 3)) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        # Position of camera in world coordinates
        self.cam_pos = self.model.cam_pos0[cam_id] + self.data.body("base_link").xpos - self.data.body("arm_base").xpos
        self.cam_init = True