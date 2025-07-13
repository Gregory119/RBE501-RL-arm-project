from os import path
import numpy as np
import mujoco
import gymnasium as gym
import copy
import time
import logging
from pprint import pformat

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dataclasses import dataclass

from lerobot.common.utils.utils import init_logging

from lerobot.common.robots import (
    RobotConfig,
    make_robot_from_config,
    so101_follower,
)
from lerobot.common.robots.so101_follower import SO101FollowerConfig, SO101Follower

from .arm import Arm


class ArmHwEnv(gym.Env):

    def __init__(self,
                 rate_hz: int = 250,
                 enable_normalize = True,
                 enable_terminate = False,
                 ):
        robot_cfg = SO101FollowerConfig(id="follower_arm",
                                        port="/dev/ttyACM0",
                                        # (radians is not an option)
                                        use_degrees=True)
        self.robot = SO101Follower(robot_cfg)

        self.robot = make_robot_from_config(robot_cfg)
        self.robot.connect()

        # load the robot model
        xml_file = path.join(
            path.dirname(__file__),
            "SO-ARM100/Simulation/SO101/scene.xml"
            )
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        get_pos_fn = self.read_pos
        get_vel_fn = self.read_vel
        load_env_fn = lambda: self.load_env()
        should_truncate_fn = lambda: not self.in_bounds()
        visualize = lambda: None

        self.observation_space = None
        def set_obs_space(obs_space):
            self.observation_space = obs_space

        self.arm = Arm(rate_hz=rate_hz,
                       get_pos_fn=get_pos_fn,
                       get_vel_fn=get_vel_fn,
                       load_env_fn=load_env_fn,
                       should_truncate_fn=should_truncate_fn,
                       vis_fn=visualize,
                       set_obs_space_fn=set_obs_space,
                       np_random=self.np_random,
                       enable_normalize=enable_normalize,
                       enable_terminate=enable_terminate)

        # Set the action space (copied from mujoco environment).
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)


    def read_pos(self):
        q_dict = self.robot.bus.sync_read("Present_Position")
        q_deg = np.array([p for _,p in q_dict.items()])
        q = q_deg / 180 * np.pi
        return q


    def read_vel(self):
        dq_dict = self.robot.bus.sync_read("Present_Velocity")
        dq_deg_p_s = np.array([v for _,v in dq_dict.items()])
        dq = dq_deg_p_s / 180 * np.pi
        return dq


    def in_bounds(self):
        q = self.read_pos()
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model,self.data)
        ee_pos = self.data.site("gripper").xpos
        return self.arm.in_bounds(ee_pos)


    def send_action(self, action):
        np_action = np.array(action) / np.pi * 180
        if not self.robot.is_connected:
            return
        keys = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
        # the unit of each action element is radian so convert to degree
        self.robot.send_action(dict(zip(keys, np_action)))


    def reset(self,
              seed: int | None = None,
              options: dict | None = None,
              ):
        super().reset(seed=seed)
        obs = self.arm.reset(mj_model=self.model, mj_data=self.data)
        info = {}
        return obs, info


    def load_env(self):
        # robot should already be connected so just reset the position
        self.send_action([0,0,0,0,0,0])

        # By default the robot doesn't use integral gain so the joint position
        # error can be large and depend on the robot configuration. Using joint
        # position errors to check for reaching the goal is therefore
        # unreliable. Instead wait for the joint velocity to reach close to
        # zero after initially moving.

        # wait for robot to start moving
        time.sleep(0.5)

        # wait for robot to reach position (stop moving)
        dq_norm_tol = 0.01
        while True:
            dq = self.read_vel()
            dq_norm = np.linalg.norm(dq)
            #print("dq_norm: {}".format(dq_norm))
            if dq_norm < dq_norm_tol:
                break
            # the wait duration doesn't have to be perfect so ignore duration of
            # sync_read()
            time.sleep(1 / self.arm.rate_hz)


    def step(self, action):
        # This needs to replicate how the simulation environment works, which
        # first steps the simulation and then sends the action. On hardware this
        # is the same as first sleeping and then sending the action.
        self.arm.step_sleep(display_rate=True)
        self.send_action(action)
        self.arm.step(action, mj_model=self.model, mj_data=self.data)


    def __del__(self):
        if hasattr(self,"robot"):
            self.robot.disconnect()
