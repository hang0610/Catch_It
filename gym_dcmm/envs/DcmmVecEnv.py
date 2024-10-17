"""
Author: Yuanhang Zhang
Version@2024-10-17
All Rights Reserved
ABOUT: this file constains the RL environment for the DCMM task
"""
import os, sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./gym_dcmm/'))
import argparse
import math
print(os.getcwd())
import configs.env.DcmmCfg as DcmmCfg
import cv2 as cv
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from gym_dcmm.agents.MujocoDcmm import MJ_DCMM
from gym_dcmm.utils.ik_pkg.ik_base import IKBase
import copy
from termcolor import colored
from decorators import *
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from utils.util import *
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from collections import deque

# os.environ['MUJOCO_GL'] = 'egl'
np.set_printoptions(precision=8)

paused = True
cmd_lin_y = 0.0
cmd_lin_x = 0.0
cmd_ang = 0.0
trigger_delta = False
trigger_delta_hand = False

def env_key_callback(keycode):
  print("chr(keycode): ", (keycode))
  global cmd_lin_y, cmd_lin_x, cmd_ang, paused, trigger_delta, trigger_delta_hand, delta_xyz, delta_xyz_hand
  if keycode == 265: # AKA: up
    cmd_lin_y += 1
    print("up %f" % cmd_lin_y)
  if keycode == 264: # AKA: down
    cmd_lin_y -= 1
    print("down %f" % cmd_lin_y)
  if keycode == 263: # AKA: left
    cmd_lin_x -= 1
    print("left: %f" % cmd_lin_x)
  if keycode == 262: # AKA: right
    cmd_lin_x += 1
    print("right %f" % cmd_lin_x) 
  if keycode == 52: # AKA: 4
    cmd_ang -= 0.2
    print("turn left %f" % cmd_ang)
  if keycode == 54: # AKA: 6
    cmd_ang += 0.2
    print("turn right %f" % cmd_ang)
  if chr(keycode) == ' ': # AKA: space
    if paused: paused = not paused
  if keycode == 334: # AKA + (on the numpad)
    trigger_delta = True
    delta_xyz = 0.1
  if keycode == 333: # AKA - (on the numpad)
    trigger_delta = True
    delta_xyz = -0.1
  if keycode == 327: # AKA 7 (on the numpad)
    trigger_delta_hand = True
    delta_xyz_hand = 0.2
  if keycode == 329: # AKA 9 (on the numpad)
    trigger_delta_hand = True
    delta_xyz_hand = -0.2

class DcmmVecEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "depth_array", "depth_rgb_array"]}
    """
    Args:
        render_mode: str
            The mode of rendering, including "rgb_array", "depth_array".
        render_per_step: bool
            Whether to render the mujoco model per simulation step.
        viewer: bool
            Whether to show the mujoco viewer.
        imshow_cam: bool
            Whether to show the camera image.
        object_eval: bool
            Use the evaluation object.
        camera_name: str
            The name of the camera.
        object_name: str
            The name of the object.
        env_time: float
            The maximum time of the environment.
        steps_per_policy: int
            The number of steps per action.
        img_size: tuple
            The size of the image.
    """
    def __init__(
        self,
        task="tracking",
        render_mode="depth_array",
        render_per_step=False,
        viewer=False,
        imshow_cam=False,
        object_eval=False,
        camera_name=["top", "wrist"],
        object_name="object",
        env_time=2.5,
        steps_per_policy=20,
        img_size=(480, 640),
        device='cuda:0',
        print_obs=False,
        print_reward=False,
        print_ctrl=False,
        print_info=False,
        print_contacts=False,
    ):
        if task not in ["Tracking", "Catching"]:
            raise ValueError("Invalid task: {}".format(task))
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.object_name = object_name
        self.imshow_cam = imshow_cam
        self.task = task
        self.img_size = img_size
        self.device = device
        self.steps_per_policy = steps_per_policy
        self.render_per_step = render_per_step
        # Print Settings
        self.print_obs = print_obs
        self.print_reward = print_reward
        self.print_ctrl = print_ctrl
        self.print_info = print_info
        self.print_contacts = print_contacts
        # Initialize the environment
        self.Dcmm = MJ_DCMM(viewer=viewer, object_name=object_name, object_eval=object_eval)
        # self.Dcmm.show_model_info()
        self.fps = 1 / (self.steps_per_policy * self.Dcmm.model.opt.timestep)
        # Randomize the Object Info
        self.random_mass = 0.25
        self.object_static_time = 0.75
        self.object_throw = False
        self.object_train = True
        if object_eval: self.set_object_eval()
        self.Dcmm.model_xml_string = self._reset_object()
        self.Dcmm.model = mujoco.MjModel.from_xml_string(self.Dcmm.model_xml_string)
        self.Dcmm.data = mujoco.MjData(self.Dcmm.model)
        # Get the geom id of the hand, the floor and the object
        self.hand_start_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'mcp_joint') - 1
        print("self.hand_start_id: ", self.hand_start_id)
        self.floor_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.object_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, self.object_name)
        self.base_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'ranger_base')

        # Set the camera configuration
        self.Dcmm.model.vis.global_.offwidth = DcmmCfg.cam_config["width"]
        self.Dcmm.model.vis.global_.offheight = DcmmCfg.cam_config["height"]
        self.mujoco_renderer = MujocoRenderer(
            self.Dcmm.model, self.Dcmm.data
        )
        if self.Dcmm.open_viewer:
            if self.Dcmm.viewer:
                print("Close the previous viewer")
                self.Dcmm.viewer.close()
            self.Dcmm.viewer = mujoco.viewer.launch_passive(self.Dcmm.model, self.Dcmm.data, key_callback=env_key_callback)
            # Modify the view position and orientation
            self.Dcmm.viewer.cam.lookat[0:2] = [0, 1]
            self.Dcmm.viewer.cam.distance = 5.0
            self.Dcmm.viewer.cam.azimuth = 180
            # self.viewer.cam.elevation = -1.57
        else: self.Dcmm.viewer = None

        # Observations are dictionaries with the agent's and the object's state. (dim = 44)
        hand_joint_indices = np.where(DcmmCfg.hand_mask == 1)[0] + 15
        self.observation_space = spaces.Dict(
            {
                "base": spaces.Dict({
                    "v_lin_2d": spaces.Box(-4, 4, shape=(2,), dtype=np.float32),
                }),
                "arm": spaces.Dict({
                    "ee_pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "ee_quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                    "ee_v_lin_3d": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                    "joint_pos": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in range(9, 15)]),
                                            high = np.array([self.Dcmm.model.jnt_range[i][1] for i in range(9, 15)]),
                                            dtype=np.float32),
                }),
                "hand": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in hand_joint_indices]),
                                   high = np.array([self.Dcmm.model.jnt_range[i][1] for i in hand_joint_indices]),
                                   dtype=np.float32),
                "object": spaces.Dict({
                    "pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "v_lin_3d": spaces.Box(-4, 4, shape=(3,), dtype=np.float32),
                    ## TODO: to be determined
                    # "shape": spaces.Box(-5, 5, shape=(2,), dtype=np.float32),
                }),
            }
        )
        # Define the limit for the mobile base action
        base_low = np.array([-4, -4])
        base_high = np.array([4, 4])
        # Define the limit for the arm action
        arm_low = -0.025*np.ones(4)
        arm_high = 0.025*np.ones(4)
        # Define the limit for the hand action
        hand_low = np.array([self.Dcmm.model.jnt_range[i][0] for i in hand_joint_indices])
        hand_high = np.array([self.Dcmm.model.jnt_range[i][1] for i in hand_joint_indices])

        # Get initial ee_pos3d
        self.init_pos = True
        self.initial_ee_pos3d = self._get_relative_ee_pos3d()
        self.initial_obj_pos3d = self._get_relative_object_pos3d()
        self.prev_ee_pos3d = np.array([0.0, 0.0, 0.0])
        self.prev_obj_pos3d = np.array([0.0, 0.0, 0.0])
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]
        self.prev_obj_pos3d[:] = self.initial_obj_pos3d[:]

        # Actions (dim = 20)
        self.action_space = spaces.Dict(
            {
                "base": spaces.Box(base_low, base_high, shape=(2,), dtype=np.float32),
                "arm": spaces.Box(arm_low, arm_high, shape=(4,), dtype=np.float32),
                "hand": spaces.Box(low = hand_low,
                                   high = hand_high,
                                   dtype = np.float32),
            }
        )
        self.action_buffer = {
            "base": DynamicDelayBuffer(maxlen=2),
            "arm": DynamicDelayBuffer(maxlen=2),
            "hand": DynamicDelayBuffer(maxlen=2),
        }
        # Combine the limits of the action space
        self.actions_low = np.concatenate([base_low, arm_low, hand_low])
        self.actions_high = np.concatenate([base_high, arm_high, hand_high])

        self.obs_dim = get_total_dimension(self.observation_space)
        self.act_dim = get_total_dimension(self.action_space)
        self.obs_t_dim = self.obs_dim - 12 - 6  # dim = 18, 12 for the hand, 6 for the arm joint positions
        self.act_t_dim = self.act_dim - 12 # dim = 6, 12 for the hand
        self.obs_c_dim = self.obs_dim - 6  # dim = 30, 6 for the arm joint positions
        self.act_c_dim = self.act_dim # dim = 18,
        print("##### Tracking Task \n obs_dim: {}, act_dim: {}".format(self.obs_t_dim, self.act_t_dim))
        print("##### Catching Task \n obs_dim: {}, act_dim: {}\n".format(self.obs_c_dim, self.act_c_dim))

        # Init env params
        self.arm_limit = True
        self.terminated = False
        self.start_time = self.Dcmm.data.time
        self.catch_time = self.Dcmm.data.time - self.start_time
        self.reward_touch = 0
        self.reward_stability = 0
        self.env_time = env_time
        self.stage_list = ["tracking", "grasping"]
        # Default stage is "tracking"
        self.stage = self.stage_list[0]
        self.steps = 0

        self.prev_ctrl = np.zeros(18)
        self.init_ctrl = True
        self.vel_init = False
        self.vel_history = deque(maxlen=4)

        self.info = {
            "ee_distance": np.linalg.norm(self.Dcmm.data.body("link6").xpos - 
                                          self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),
            "base_distance": np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] - 
                                            self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),
            "env_time": self.Dcmm.data.time - self.start_time,
            "imgs": {}
        }
        self.contacts = {
            # Get contact point from the mujoco model
            "object_contacts": np.array([]),
            "hand_contacts": np.array([]),
        }

        self.object_q = np.array([1, 0, 0, 0])
        self.object_pos3d = np.array([0, 0, 1.5])
        self.object_vel6d = np.array([0., 0., 1.25, 0.0, 0.0, 0.0])
        self.step_touch = False

        self.imgs = np.zeros((0, self.img_size[0], self.img_size[1], 1))

        # Random PID Params
        self.k_arm = np.ones(6)
        self.k_drive = np.ones(4)
        self.k_steer = np.ones(4)
        self.k_hand = np.ones(1)
        # Random Obs & Act Params
        self.k_obs_base = DcmmCfg.k_obs_base
        self.k_obs_arm = DcmmCfg.k_obs_arm
        self.k_obs_hand = DcmmCfg.k_obs_hand
        self.k_obs_object = DcmmCfg.k_obs_object
        self.k_act = DcmmCfg.k_act

    def set_object_eval(self):
        self.object_train = False

    def update_render_state(self, render_per_step):
        self.render_per_step = render_per_step

    def update_stage(self, stage):
        if stage in self.stage_list:
            self.stage = stage
        else:
            raise ValueError("Invalid stage: {}".format(stage))

    def _get_contacts(self):
        # Contact information of the hand
        geom_ids = self.Dcmm.data.contact.geom
        geom1_ids = self.Dcmm.data.contact.geom1
        geom2_ids = self.Dcmm.data.contact.geom2
        ## get the contact points of the hand
        geom1_hand = np.where((geom1_ids < self.object_id) & (geom1_ids >= self.hand_start_id))[0]
        geom2_hand = np.where((geom2_ids < self.object_id) & (geom2_ids >= self.hand_start_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_hand.size != 0:
            contacts_geom1 = geom_ids[geom1_hand][:,1]
        if geom2_hand.size != 0:
            contacts_geom2 = geom_ids[geom2_hand][:,0]
        hand_contacts = np.concatenate((contacts_geom1, contacts_geom2))
        ## get the contact points of the object
        geom1_object = np.where((geom1_ids == self.object_id))[0]
        geom2_object = np.where((geom2_ids == self.object_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_object.size != 0:
            contacts_geom1 = geom_ids[geom1_object][:,1]
        if geom2_object.size != 0:
            contacts_geom2 = geom_ids[geom2_object][:,0]
        object_contacts = np.concatenate((contacts_geom1, contacts_geom2))
        ## get the contact points of the base
        geom1_base = np.where((geom1_ids == self.base_id))[0]
        geom2_base = np.where((geom2_ids == self.base_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_base.size != 0:
            contacts_geom1 = geom_ids[geom1_base][:,1]
        if geom2_base.size != 0:
            contacts_geom2 = geom_ids[geom2_base][:,0]
        base_contacts = np.concatenate((contacts_geom1, contacts_geom2))
        if self.print_contacts:
            print("object_contacts: ", object_contacts)
            print("hand_contacts: ", hand_contacts)
            print("base_contacts: ", base_contacts)
        return {
            # Get contact point from the mujoco model
            "object_contacts": object_contacts,
            "hand_contacts": hand_contacts,
            "base_contacts": base_contacts
        }

    def _get_base_vel(self):
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        global_base_vel = self.Dcmm.data.qvel[0:2]
        base_vel_x = math.cos(base_yaw) * global_base_vel[0] + math.sin(base_yaw) * global_base_vel[1]
        base_vel_y = -math.sin(base_yaw) * global_base_vel[0] + math.cos(base_yaw) * global_base_vel[1]
        return np.array([base_vel_x, base_vel_y])

    def _get_relative_ee_pos3d(self):
        # Caclulate the ee_pos3d w.r.t. the base_link
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        x,y = relative_position(self.Dcmm.data.body("arm_base").xpos[0:2], 
                                self.Dcmm.data.body("link6").xpos[0:2], 
                                base_yaw)
        return np.array([x, y, 
                         self.Dcmm.data.body("link6").xpos[2]-self.Dcmm.data.body("arm_base").xpos[2]])

    def _get_relative_ee_quat(self):
        # Caclulate the ee_pos3d w.r.t. the base_link
        quat = relative_quaternion(self.Dcmm.data.body("base_link").xquat, self.Dcmm.data.body("link6").xquat)
        return np.array(quat)

    def _get_relative_ee_v_lin_3d(self):
        # Caclulate the ee_v_lin3d w.r.t. the base_link
        # In simulation, we can directly get the velocity of the end-effector
        base_vel = self.Dcmm.data.body("arm_base").cvel[3:6]
        global_ee_v_lin = self.Dcmm.data.body("link6").cvel[3:6]
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        ee_v_lin_x = math.cos(base_yaw) * (global_ee_v_lin[0]-base_vel[0]) + math.sin(base_yaw) * (global_ee_v_lin[1]-base_vel[1])
        ee_v_lin_y = -math.sin(base_yaw) * (global_ee_v_lin[0]-base_vel[0]) + math.cos(base_yaw) * (global_ee_v_lin[1]-base_vel[1])
        # TODO: In the real world, we can only estimate it by differentiating the position
        return np.array([ee_v_lin_x, ee_v_lin_y, global_ee_v_lin[2]-base_vel[2]])
    
    def _get_relative_object_pos3d(self):
        # Caclulate the object_pos3d w.r.t. the base_link
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        x,y = relative_position(self.Dcmm.data.body("arm_base").xpos[0:2], 
                                self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2], 
                                base_yaw)
        return np.array([x, y, 
                         self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]-self.Dcmm.data.body("arm_base").xpos[2]])

    def _get_relative_object_v_lin_3d(self):
        # Caclulate the object_v_lin3d w.r.t. the base_link
        base_vel = self.Dcmm.data.body("arm_base").cvel[3:6]
        global_object_v_lin = self.Dcmm.data.joint(self.Dcmm.object_name).qvel[0:3]
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        object_v_lin_x = math.cos(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.sin(base_yaw) * (global_object_v_lin[1]-base_vel[1])
        object_v_lin_y = -math.sin(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.cos(base_yaw) * (global_object_v_lin[1]-base_vel[1])
        return np.array([object_v_lin_x, object_v_lin_y, global_object_v_lin[2]-base_vel[2]])

    def _get_obs(self):
        ee_pos3d = self._get_relative_ee_pos3d()
        obj_pos3d = self._get_relative_object_pos3d()
        if self.init_pos:
            self.prev_ee_pos3d[:] = ee_pos3d[:]
            self.prev_obj_pos3d[:] = obj_pos3d[:]
            self.init_pos = False
        # Add Obs Noise (Additive self.k_obs_base/arm/hand/object)
        obs = {
            "base": {
                "v_lin_2d": self._get_base_vel() + np.random.normal(0, self.k_obs_base, 2),
            },
            "arm": {
                "ee_pos3d": ee_pos3d + np.random.normal(0, self.k_obs_arm, 3),
                "ee_quat": self._get_relative_ee_quat() + np.random.normal(0, self.k_obs_arm, 4),
                'ee_v_lin_3d': (ee_pos3d - self.prev_ee_pos3d)*self.fps + np.random.normal(0, self.k_obs_arm, 3),
                "joint_pos": np.array(self.Dcmm.data.qpos[15:21]) + np.random.normal(0, self.k_obs_arm, 6),
            },
            "hand": self._get_hand_obs() + np.random.normal(0, self.k_obs_hand, 12),
            "object": {
                "pos3d": obj_pos3d + np.random.normal(0, self.k_obs_object, 3),
                # "v_lin_3d": self._get_relative_object_v_lin_3d() + np.random.normal(0, self.k_obs_object, 3),
                "v_lin_3d": (obj_pos3d - self.prev_obj_pos3d)*self.fps + np.random.normal(0, self.k_obs_object, 3),
            },
        }
        self.prev_ee_pos3d = ee_pos3d
        self.prev_obj_pos3d = obj_pos3d
        if self.print_obs:
            print("##### print obs: \n", obs)
        return obs
        # return obs_tensor

    def _get_hand_obs(self):
        # print("full hand: ", self.Dcmm.data.qpos[21:37])
        hand_obs = np.zeros(12)
        # Thumb
        hand_obs[9] = self.Dcmm.data.qpos[21+13]
        hand_obs[10] = self.Dcmm.data.qpos[21+14]
        hand_obs[11] = self.Dcmm.data.qpos[21+15]
        # Other Three Fingers
        hand_obs[0] = self.Dcmm.data.qpos[21]
        hand_obs[1:3] = self.Dcmm.data.qpos[(21+2):(21+4)]
        hand_obs[3] = self.Dcmm.data.qpos[21+4]
        hand_obs[4:6] = self.Dcmm.data.qpos[(21+6):(21+8)]
        hand_obs[6] = self.Dcmm.data.qpos[21+8]
        hand_obs[7:9] = self.Dcmm.data.qpos[(21+10):(21+12)]
        return hand_obs
    
    def _get_info(self):
        # Time of the Mujoco environment
        env_time = self.Dcmm.data.time - self.start_time
        ee_distance = np.linalg.norm(self.Dcmm.data.body("link6").xpos - 
                                    self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3])
        base_distance = np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                        self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2])
        # print("base_distance: ", base_distance)
        if self.print_info: 
            print("##### print info")
            print("env_time: ", env_time)
            print("ee_distance: ", ee_distance)
        return {
            # Get contact point from the mujoco model
            "env_time": env_time,
            "ee_distance": ee_distance,
            "base_distance": base_distance,
        }
    
    def update_target_ctrl(self):
        self.action_buffer["base"].append(copy.deepcopy(self.Dcmm.target_base_vel[:]))
        self.action_buffer["arm"].append(copy.deepcopy(self.Dcmm.target_arm_qpos[:]))
        self.action_buffer["hand"].append(copy.deepcopy(self.Dcmm.target_hand_qpos[:]))

    def _get_ctrl(self):
        # Map the action to the control 
        mv_steer, mv_drive = self.Dcmm.move_base_vel(self.action_buffer["base"][0]) # 8
        mv_arm = self.Dcmm.arm_pid.update(self.action_buffer["arm"][0], self.Dcmm.data.qpos[15:21], self.Dcmm.data.time) # 6
        mv_hand = self.Dcmm.hand_pid.update(self.action_buffer["hand"][0], self.Dcmm.data.qpos[21:37], self.Dcmm.data.time) # 16
        ctrl = np.concatenate([mv_steer, mv_drive, mv_arm, mv_hand], axis=0)
        # Add Action Noise (Scale with self.k_act)
        ctrl *= np.random.normal(1, self.k_act, 30)
        if self.print_ctrl:
            print("##### ctrl:")
            print("mv_steer: {}, \nmv_drive: {}, \nmv_arm: {}, \nmv_hand: {}\n".format(mv_steer, mv_drive, mv_arm, mv_hand))
        return ctrl

    def _reset_object(self):
        # Parse the XML string
        root = ET.fromstring(self.Dcmm.model_xml_string)

        # Find the <body> element with name="object"
        object_body = root.find(".//body[@name='object']")
        if object_body is not None:
            inertial = object_body.find("inertial")
            if inertial is not None:
                # Generate a random mass within the specified range
                self.random_mass = np.random.uniform(DcmmCfg.object_mass[0], DcmmCfg.object_mass[0])
                # Update the mass attribute
                inertial.set("mass", str(self.random_mass))
            joint = object_body.find("joint")
            if joint is not None:
                # Generate a random damping within the specified range
                random_damping = np.random.uniform(DcmmCfg.object_damping[0], DcmmCfg.object_damping[1])
                # Update the damping attribute
                joint.set("damping", str(random_damping))
            # Find the <geom> element
            geom = object_body.find(".//geom[@name='object']")
            if geom is not None:
                # Modify the type and size attributes
                object_id = np.random.choice([0, 1, 2, 3, 4])
                if self.object_train:
                    object_shape = DcmmCfg.object_shape[object_id]
                    geom.set("type", object_shape)  # Replace "box" with the desired type
                    object_size = np.array([np.random.uniform(low=low, high=high) for low, high in DcmmCfg.object_size[object_shape]])
                    geom.set("size", np.array_str(object_size)[1:-1])  # Replace with the desired size
                    # print("### Object Geom Info ###")
                    # for key, value in geom.attrib.items():
                    #     print(f"{key}: {value}")
                else:
                    object_mesh = DcmmCfg.object_mesh[object_id]
                    geom.set("mesh", object_mesh)
        # Convert the XML element tree to a string
        xml_str = ET.tostring(root, encoding='unicode')
        
        return xml_str

    def random_object_pose(self):
        # Random Position
        x = np.random.rand() - 0.5 # (-0.5, 0.5)
        y = 2.2 + 0.3 * np.random.rand() # (2.2, 2.5)
        # Low or High Starting Position
        low_factor = False if np.random.rand() < 0.5 else True
        # low_factor = True
        if low_factor: height = 0.7 + 0.3 * np.random.rand()# (0.7, 1.0)
        else: height = 1.0 + 0.6 * np.random.rand() # (1.0, 1.6)
        # Random Velocity
        r_vel = 1 + np.random.rand() # (1, 2)
        alpha_vel = math.pi * (np.random.rand()*1/6 + 5/12) # alpha_vel = (5/12 * pi, 7/12 * pi)
        # alpha_vel = math.pi * (np.random.rand()*1/3 + 1/3) # alpha_vel = (1/3 * pi, 2/3 * pi)
        v_lin_x = r_vel * math.cos(alpha_vel) # (-0.0, -0.5)
        v_lin_y = - r_vel * math.sin(alpha_vel) # (-0.5, -1.0)
        v_lin_z = 0.5 * np.random.rand() + 2.0 # (2.0, 2.5)
        if y > 2.25: v_lin_y -= 0.4
        if height < 1.0: v_lin_z += 1
        self.object_pos3d = np.array([x, y, height])
        self.object_vel6d = np.array([v_lin_x, v_lin_y, v_lin_z, 0.0, 0.0, 0.0])
        # Random Static Time
        self.object_static_time = np.random.uniform(DcmmCfg.object_static[0], DcmmCfg.object_static[1])
        # Random Quaternion
        r_obj_quat = R.from_euler('xyz', [0, np.random.rand()*1*math.pi, 0], degrees=False)
        self.object_q = r_obj_quat.as_quat()

    
    def random_PID(self):
        # Random the PID Controller Params in DCMM
        self.k_arm = np.random.uniform(0, 1, size=6)
        self.k_drive = np.random.uniform(0, 1, size=4)
        self.k_steer = np.random.uniform(0, 1, size=4)
        self.k_hand = np.random.uniform(0, 1, size=1)
        # Reset the PID Controller
        self.Dcmm.arm_pid.reset(self.k_arm*(DcmmCfg.k_arm[1]-DcmmCfg.k_arm[0])+DcmmCfg.k_arm[0])
        self.Dcmm.steer_pid.reset(self.k_steer*(DcmmCfg.k_steer[1]-DcmmCfg.k_steer[0])+DcmmCfg.k_steer[0])
        self.Dcmm.drive_pid.reset(self.k_drive*(DcmmCfg.k_drive[1]-DcmmCfg.k_drive[0])+DcmmCfg.k_drive[0])
        self.Dcmm.hand_pid.reset(self.k_hand[0]*(DcmmCfg.k_hand[1]-DcmmCfg.k_hand[0])+DcmmCfg.k_hand[0])
    
    def random_delay(self):
        # Random the Delay Buffer Params in DCMM
        self.action_buffer["base"].set_maxlen(np.random.choice(DcmmCfg.act_delay['base']))
        self.action_buffer["arm"].set_maxlen(np.random.choice(DcmmCfg.act_delay['arm']))
        self.action_buffer["hand"].set_maxlen(np.random.choice(DcmmCfg.act_delay['hand']))
        # Clear Buffer
        self.action_buffer["base"].clear()
        self.action_buffer["arm"].clear()
        self.action_buffer["hand"].clear()

    def _reset_simulation(self):
        # Reset the data in Mujoco Simulation
        mujoco.mj_resetData(self.Dcmm.model, self.Dcmm.data)
        mujoco.mj_resetData(self.Dcmm.model_arm, self.Dcmm.data_arm)
        if self.Dcmm.model.na == 0:
            self.Dcmm.data.act[:] = None
        if self.Dcmm.model_arm.na == 0:
            self.Dcmm.data_arm.act[:] = None
        self.Dcmm.data.ctrl = np.zeros(self.Dcmm.model.nu)
        self.Dcmm.data_arm.ctrl = np.zeros(self.Dcmm.model_arm.nu)
        self.Dcmm.data.qpos[15:21] = DcmmCfg.arm_joints[:]
        self.Dcmm.data.qpos[21:37] = DcmmCfg.hand_joints[:]
        self.Dcmm.data_arm.qpos[0:6] = DcmmCfg.arm_joints[:]
        self.Dcmm.data.body("object").xpos[0:3] = np.array([2, 2, 1])
        # Random 3D position TODO: Adjust to the fov
        self.random_object_pose()
        self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                    velocity=np.zeros(6))
        # TODO: TESTING
        # self.Dcmm.set_throw_pos_vel(pose=np.array([0.0, 0.4, 1.0, 1.0, 0.0, 0.0, 0.0]),
        #                             velocity=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        # Random Gravity
        self.Dcmm.model.opt.gravity[2] = -9.81 + 0.5*np.random.uniform(-1, 1)
        # Random PID
        self.random_PID()
        # Random Delay
        self.random_delay()
        # Forward Kinematics
        mujoco.mj_forward(self.Dcmm.model, self.Dcmm.data)
        mujoco.mj_forward(self.Dcmm.model_arm, self.Dcmm.data_arm)

    def reset(self):
        # Reset the basic simulation
        self._reset_simulation()
        self.init_ctrl = True
        self.init_pos = True
        self.vel_init = False
        self.object_throw = False
        self.steps = 0
        # Reset the time
        self.start_time = self.Dcmm.data.time
        self.catch_time = self.Dcmm.data.time - self.start_time

        ## Reset the target velocity of the mobile base
        self.Dcmm.target_base_vel = np.array([0.0, 0.0, 0.0])
        ## Reset the target joint positions of the arm
        self.Dcmm.target_arm_qpos[:] = DcmmCfg.arm_joints[:]
        ## Reset the target joint positions of the hand
        self.Dcmm.target_hand_qpos[:] = DcmmCfg.hand_joints[:]
        ## Reset the reward
        self.stage = "tracking"
        self.terminated = False
        self.reward_touch = 0
        self.reward_stability = 0

        self.info = {
            "ee_distance": np.linalg.norm(self.Dcmm.data.body("link6").xpos - 
                                       self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),
            "base_distance": np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),
            "evn_time": self.Dcmm.data.time - self.start_time,
        }
        # Get the observation and info
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]
        self.prev_obj_pos3d = self._get_relative_object_pos3d()
        observation = self._get_obs()
        info = self._get_info()
        # Rendering
        imgs = self.render()
        info['imgs'] = imgs
        ctrl_delay = np.array([len(self.action_buffer['base']),
                               len(self.action_buffer['arm']),
                               len(self.action_buffer['hand'])])
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_drive, self.k_hand, ctrl_delay))

        return observation, info

    def norm_ctrl(self, ctrl, components):
        '''
        Convert the ctrl (dict type) to the numpy array and return its norm value
        Input: ctrl, dict
        Return: norm, float
        '''
        ctrl_array = np.concatenate([ctrl[component]*DcmmCfg.reward_weights['r_ctrl'][component] for component in components])
        return np.linalg.norm(ctrl_array)

    def compute_reward(self, obs, info, ctrl):
        '''
        Rewards:
        - Object Position Reward
        - Object Orientation Reward
        - Object Touch Success Reward
        - Object Catch Stability Reward
        - Collision Penalty
        - Constraint Penalty
        '''
        rewards = 0.0
        ## Object Position Reward (-inf, 0)
        # Compute the closest distance the end-effector comes to the object
        reward_base_pos = (self.info["base_distance"] - info["base_distance"]) * DcmmCfg.reward_weights["r_base_pos"]
        reward_ee_pos = (self.info["ee_distance"] - info["ee_distance"]) * DcmmCfg.reward_weights["r_ee_pos"]
        reward_ee_precision = math.exp(-50*info["ee_distance"]**2) * DcmmCfg.reward_weights["r_precision"]

        ## Collision Penalty
        # Compute the Penalty when the arm is collided with the mobile base
        reward_collision = 0
        if self.contacts['base_contacts'].size != 0:
            reward_collision = DcmmCfg.reward_weights["r_collision"]
        
        ## Constraint Penalty
        # Compute the Penalty when the arm joint position is out of the joint limits
        reward_constraint = 0 if self.arm_limit else -1
        reward_constraint *= DcmmCfg.reward_weights["r_constraint"]

        ## Object Touch Success Reward
        # Compute the reward when the object is caught successfully by the hand
        if self.step_touch:
            # print("TRACK SUCCESS!!!!!")
            if not self.reward_touch:
                self.catch_time = self.Dcmm.data.time - self.start_time
            self.reward_touch = DcmmCfg.reward_weights["r_touch"][self.task]
        else:
            self.reward_touch = 0

        if self.task == "Catching":
            reward_orient = 0
            ## Calculate the total reward in different stages
            if self.stage == "tracking":
                ## Ctrl Penalty
                # Compute the norm of hand joint movement through the current actions in the tracking stage
                reward_ctrl = - self.norm_ctrl(ctrl, {"hand"})
                ## Object Orientation Reward
                # Compute the dot product of the velocity vector of the object and the z axis of the end_effector
                rotation_matrix = quaternion_to_rotation_matrix(obs["arm"]["ee_quat"])
                local_velocity_vector = np.dot(rotation_matrix.T, obs["object"]["v_lin_3d"])
                hand_z_axis = np.array([0, 0, 1])
                reward_orient = abs(cos_angle_between_vectors(local_velocity_vector, hand_z_axis)) * DcmmCfg.reward_weights["r_orient"]
                ## Add up the rewards
                rewards = reward_base_pos + reward_ee_pos + reward_orient + reward_ctrl + reward_collision + reward_constraint + self.reward_touch
                if self.print_reward:
                    if reward_constraint < 0:
                        print("ctrl: ", ctrl)
                    print("### print reward")
                    print("reward_ee_pos: {:.3f}, reward_ee_precision: {:.3f}, reward_orient: {:.3f}, reward_ctrl: {:.3f}, \n".format(
                        reward_ee_pos, reward_ee_precision, reward_orient, reward_ctrl
                    ) + "reward_collision: {:.3f}, reward_constraint: {:.3f}, reward_touch: {:.3f}".format(
                        reward_collision, reward_constraint, self.reward_touch
                    ))
                    print("total reward: {:.3f}\n".format(rewards))
            else:
                ## Ctrl Penalty
                # Compute the norm of base and arm movement through the current actions in the grasping stage
                reward_ctrl = - self.norm_ctrl(ctrl, {"base", "arm"})
                ## Set the Orientation Reward to maximum (1)
                reward_orient = 1
                ## Object Touch Stability Reward
                # Compute the reward when the object is caught stably in the hand
                if self.reward_touch:
                    self.reward_stability = (info["env_time"] - self.catch_time) * DcmmCfg.reward_weights["r_stability"]
                else:
                    self.reward_stability = 0.0
                ## Add up the rewards
                rewards = reward_base_pos + reward_ee_pos + reward_ee_precision + reward_orient + reward_ctrl + reward_collision + reward_constraint \
                        + self.reward_touch + self.reward_stability
                if self.print_reward:
                    print("##### print reward")
                    print("reward_touch: {}, \nreward_ee_pos: {:.3f}, reward_ee_precision: {:.3f}, reward_orient: {:.3f}, \n".format(
                        self.reward_touch, reward_ee_pos, reward_ee_precision, reward_orient
                    ) + "reward_stability: {:.3f}, reward_collision: {:.3f}, \nreward_ctrl: {:.3f}, reward_constraint: {:.3f}".format(
                        self.reward_stability, reward_collision, reward_ctrl, reward_constraint
                    ))
                    print("total reward: {:.3f}\n".format(rewards))
        elif self.task == 'Tracking':
            ## Ctrl Penalty
            # Compute the norm of base and arm movement through the current actions in the grasping stage
            reward_ctrl = - self.norm_ctrl(ctrl, {"base", "arm"})
            ## Object Orientation Reward
            # Compute the dot product of the velocity vector of the object and the z axis of the end_effector
            rotation_matrix = quaternion_to_rotation_matrix(obs["arm"]["ee_quat"])
            local_velocity_vector = np.dot(rotation_matrix.T, obs["object"]["v_lin_3d"])
            hand_z_axis = np.array([0, 0, 1])
            reward_orient = abs(cos_angle_between_vectors(local_velocity_vector, hand_z_axis)) * DcmmCfg.reward_weights["r_orient"]
            ## Add up the rewards
            rewards = reward_base_pos + reward_ee_pos + reward_ee_precision + reward_orient + reward_ctrl + reward_collision + reward_constraint + self.reward_touch
            if self.print_reward:
                if reward_constraint < 0:
                    print("ctrl: ", ctrl)
                print("### print reward")
                print("reward_ee_pos: {:.3f}, reward_ee_precision: {:.3f}, reward_orient: {:.3f}, reward_ctrl: {:.3f}, \n".format(
                    reward_ee_pos, reward_ee_precision, reward_orient, reward_ctrl
                ) + "reward_collision: {:.3f}, reward_constraint: {:.3f}, reward_touch: {:.3f}".format(
                    reward_collision, reward_constraint, self.reward_touch
                ))
                print("total reward: {:.3f}\n".format(rewards))
        else:
            raise ValueError("Invalid task: {}".format(self.task))
        
        return rewards

    def _step_mujoco_simulation(self, action_dict):
        ## TODO: Low-Pass-Filter the Base Velocity
        self.Dcmm.target_base_vel[0:2] = action_dict['base']
        action_arm = np.concatenate((action_dict["arm"], np.zeros(3)))
        result_QP, _ = self.Dcmm.move_ee_pose(action_arm)
        if result_QP[1]:
            self.arm_limit = True
            self.Dcmm.target_arm_qpos[:] = result_QP[0]
        else:
            # print("IK Failed!!!")
            self.arm_limit = False
        self.Dcmm.action_hand2qpos(action_dict["hand"])
        # Add Target Action to the Buffer
        self.update_target_ctrl()
        # Reset the Criteria for Successfully Touch
        self.step_touch = False
        for _ in range(self.steps_per_policy):
            # Update the control command according to the latest policy output
            self.Dcmm.data.ctrl[:-1] = self._get_ctrl()
            if self.render_per_step:
                # Rendering
                img = self.render()
            # As of MuJoCo 2.0, force-related quantities like cacc are not computed
            # unless there's a force sensor in the model.
            # See https://github.com/openai/gym/issues/1541
            if self.Dcmm.data.time - self.start_time < self.object_static_time:
                self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                            velocity=np.zeros(6))
                self.Dcmm.data.ctrl[-1] = self.random_mass * -self.Dcmm.model.opt.gravity[2]
            elif not self.object_throw:
                self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                            velocity=self.object_vel6d[:])
                self.Dcmm.data.ctrl[-1] = 0.0
                self.object_throw = True

            mujoco.mj_step(self.Dcmm.model, self.Dcmm.data)
            mujoco.mj_rnePostConstraint(self.Dcmm.model, self.Dcmm.data)

            # Update the contact information
            self.contacts = self._get_contacts()
            # Whether the base collides
            if self.contacts['base_contacts'].size != 0:
                self.terminated = True
            mask_coll = self.contacts['object_contacts'] < self.hand_start_id
            mask_finger = self.contacts['object_contacts'] > self.hand_start_id
            mask_hand = self.contacts['object_contacts'] >= self.hand_start_id
            mask_palm = self.contacts['object_contacts'] == self.hand_start_id
            # Whether the object is caught
            if self.step_touch == False:
                if self.task == "Catching" and np.any(mask_hand):
                    self.step_touch = True
                elif self.task == "Tracking" and np.any(mask_palm):
                    self.step_touch = True
            # Whether the object falls
            if not self.terminated:
                if self.task == "Catching":
                    self.terminated = np.any(mask_coll)
                elif self.task == "Tracking":
                    self.terminated = np.any(mask_coll) or np.any(mask_finger)
            # If the object falls, terminate the episode in advance
            if self.terminated:
                break

    def step(self, action):
        self.steps += 1
        self._step_mujoco_simulation(action)
        # Get the obs and info
        obs = self._get_obs()
        info = self._get_info()
        if self.task == 'Catching':
            if info['ee_distance'] < DcmmCfg.distance_thresh and self.stage == "tracking":
                self.stage = "grasping"
            elif info['ee_distance'] >= DcmmCfg.distance_thresh and self.stage == "grasping":
                self.terminated = True
        # Design the reward function
        reward = self.compute_reward(obs, info, action)
        self.info["base_distance"] = info["base_distance"]
        self.info["ee_distance"] = info["ee_distance"]
        # Rendering
        imgs = self.render()
        # Update the imgs
        info['imgs'] = imgs
        ctrl_delay = np.array([len(self.action_buffer['base']),
                               len(self.action_buffer['arm']),
                               len(self.action_buffer['hand'])])
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_drive, self.k_hand, ctrl_delay))
        # The episode is truncated if the env_time is larger than the predefined time
        if self.task == "Catching":
            if info["env_time"] > self.env_time:
                # print("Catching Success!!!!!!")
                truncated = True
            else: truncated = False
        elif self.task == "Tracking":
            if self.step_touch:
                # print("Tracking Success!!!!!!")
                truncated = True
            else: truncated = False
        terminated = self.terminated
        done = terminated or truncated
        if done:
            # TEST ONLY
            # self.reset()
            pass
        return obs, reward, terminated, truncated, info

    def preprocess_depth_with_mask(self, rgb_img, depth_img, 
                                   depth_threshold=3.0, 
                                   num_white_points_range=(5, 15),
                                   point_size_range=(1, 5)):
        # Define RGB Filter
        lower_rgb = np.array([5, 0, 0])
        upper_rgb = np.array([255, 15, 15])
        rgb_mask = cv.inRange(rgb_img, lower_rgb, upper_rgb)
        depth_mask = cv.inRange(depth_img, 0, depth_threshold)
        combined_mask = np.logical_and(rgb_mask, depth_mask)
        # Apply combined mask to depth image
        masked_depth_img = np.where(combined_mask, depth_img, 0)
        # Calculate mean depth within combined mask
        masked_depth_mean = np.nanmean(np.where(combined_mask, depth_img, np.nan))
        # Generate random number of white points
        num_white_points = np.random.randint(num_white_points_range[0], num_white_points_range[1])
        # Generate random coordinates for white points
        random_x = np.random.randint(0, depth_img.shape[1], size=num_white_points)
        random_y = np.random.randint(0, depth_img.shape[0], size=num_white_points)
        # Generate random sizes for white points in the specified range
        random_sizes = np.random.randint(point_size_range[0], point_size_range[1], size=num_white_points)
        # Create masks for all white points at once
        y, x = np.ogrid[:masked_depth_img.shape[0], :masked_depth_img.shape[1]]
        point_masks = ((x[..., None] - random_x) ** 2 + (y[..., None] - random_y) ** 2) <= random_sizes ** 2
        # Update masked depth image with the white points
        masked_depth_img[np.any(point_masks, axis=2)] = np.random.uniform(1.5, 3.0)

        return masked_depth_img, masked_depth_mean

    def render(self):
        imgs = np.zeros((0, self.img_size[0], self.img_size[1]))
        imgs_depth = np.zeros((0, self.img_size[0], self.img_size[1]))
        # imgs_rgb = np.zeros((self.img_size[0], self.img_size[1], 3))
        for camera_name in self.camera_name:
            if self.render_mode == "human":
                self.mujoco_renderer.render(
                    self.render_mode, camera_name = camera_name
                )
                return imgs
            elif self.render_mode != "depth_rgb_array":
                img = self.mujoco_renderer.render(
                    self.render_mode, camera_name = camera_name
                )
                if self.imshow_cam and self.render_mode == "rgb_array":
                    cv.imshow(camera_name, cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    cv.waitKey(1)
                # Converts the depth array valued from 0-1 to real meters
                elif self.render_mode == "depth_array":
                    img = self.Dcmm.depth_2_meters(img)
                    if self.imshow_cam:
                        depth_norm = np.zeros(img.shape, dtype=np.uint8)
                        cv.convertScaleAbs(img, depth_norm, alpha=(255.0/img.max()))
                        cv.imshow(camera_name+"_depth", depth_norm)
                        cv.waitKey(1)
                    img = np.expand_dims(img, axis=0)
            else:
                img_rgb = self.mujoco_renderer.render(
                    "rgb_array", camera_name = camera_name
                )
                img_depth = self.mujoco_renderer.render(
                    "depth_array", camera_name = camera_name
                )   
                # Converts the depth array valued from 0-1 to real meters
                img_depth = self.Dcmm.depth_2_meters(img_depth)
                img_depth, _ = self.preprocess_depth_with_mask(img_rgb, img_depth)
                if self.imshow_cam:
                    cv.imshow(camera_name+"_rgb", cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
                    cv.imshow(camera_name+"_depth", img_depth)
                    cv.waitKey(1)
                img_depth = cv.resize(img_depth, (self.img_size[1], self.img_size[0]))
                img_depth = np.expand_dims(img_depth, axis=0)
                imgs_depth = np.concatenate((imgs_depth, img_depth), axis=0)
            # Sync the viewer (if exists) with the data
            if self.Dcmm.viewer != None: 
                self.Dcmm.viewer.sync()
        if self.render_mode == "depth_rgb_array":
            # Only keep the depth image
            imgs = imgs_depth
        return imgs

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
        if self.Dcmm.viewer != None: self.Dcmm.viewer.close()

    def run_test(self):
        global cmd_lin_x, cmd_lin_y, trigger_delta, trigger_delta_hand, delta_xyz, delta_xyz_hand
        self.reset()
        action = np.zeros(18)
        while True:
            # Note: action's dim = 18, which includes 2 for the base, 4 for the arm, and 12 for the hand
            # print("##### stage: ", self.stage)
            # Keyboard control
            action[0:2] = np.array([cmd_lin_x, cmd_lin_y])
            if trigger_delta:
                print("delta_xyz: ", delta_xyz)
                action[2:6] = np.array([delta_xyz, delta_xyz, delta_xyz, delta_xyz])
                trigger_delta = False
            else:
                action[2:6] = np.zeros(4)
            if trigger_delta_hand:
                print("delta_xyz_hand: ", delta_xyz_hand)
                action[6:18] = np.ones(12)*delta_xyz_hand
                trigger_delta_hand = False
            else:
                action[6:18] = np.zeros(12)
            base_tensor = action[:2]
            arm_tensor = action[2:6]
            hand_tensor = action[6:18]
            actions_dict = {
                'arm': arm_tensor,
                'base': base_tensor,
                'hand': hand_tensor
            }
            # print("self.Dcmm.data.body('link6'):", self.Dcmm.data.body('link6'))
            observation, reward, terminated, truncated, info = self.step(actions_dict)

if __name__ == "__main__":
    os.chdir('../../')
    parser = argparse.ArgumentParser(description="Args for DcmmVecEnv")
    parser.add_argument('--viewer', action='store_true', help="open the mujoco.viewer or not")
    parser.add_argument('--imshow_cam', action='store_true', help="imshow the camera image or not")
    args = parser.parse_args()
    print("args: ", args)
    env = DcmmVecEnv(task='Catching', object_name='object', render_per_step=False, 
                    print_reward=False, print_info=False, 
                    print_contacts=False, print_ctrl=False, 
                    print_obs=False, camera_name = ["top"],
                    render_mode="rgb_array", imshow_cam=args.imshow_cam, 
                    viewer = args.viewer, object_eval=False,
                    env_time = 2.5, steps_per_policy=20)
    env.run_test()