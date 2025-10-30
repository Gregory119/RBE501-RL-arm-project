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
from .arm import xyz_to_rpz, rpz_to_xyz


class ArmHwEnv(gym.Env):

    def __init__(self,
                 rate_hz: int = 50,
                 enable_normalize = True,
                 enable_terminate = False,
                 enable_random_goal = True,
                 rpz_low = None,
                 rpz_high = None,
                 ):
        robot_cfg = SO101FollowerConfig(id="rl_follower_arm",
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

        get_qpos_fn = self.read_pos
        get_qvel_fn = self.read_vel
        load_env_fn = lambda: self.load_env()
        should_truncate_fn = lambda q: not self.in_bounds(q)
        visualize = lambda: None

        self.observation_space = None
        def set_obs_space(obs_space):
            self.observation_space = obs_space

        default_goal_rpz = None
        if not enable_random_goal:
            rho = 0.0254*12
            phi = -np.pi/2
            z = 0.0254*15
            default_goal_rpz = (rho,phi,z)

        self.arm = Arm(rate_hz=rate_hz,
                       get_qpos_fn=get_qpos_fn,
                       get_qvel_fn=get_qvel_fn,
                       load_env_fn=load_env_fn,
                       should_truncate_fn=should_truncate_fn,
                       vis_fn=visualize,
                       set_obs_space_fn=set_obs_space,
                       np_random=self.np_random,
                       enable_normalize=enable_normalize,
                       enable_terminate=enable_terminate,
                       rpz_low=rpz_low,
                       rpz_high=rpz_high,
                       assert_obs=False,
                       default_goal_rpz=default_goal_rpz)

        # Set the action space to scale relative position
        self.action_space = Box(low=-1, high=1, shape=(6,), dtype=np.float64)

        self.read_pos_cnt = 0
        self.read_vel_cnt = 0


    def read_pos(self):
        self.read_pos_cnt += 1
        failed_once = False
        while True:
            try:
                q_dict = self.robot.bus.sync_read("Present_Position")
                q_deg = np.array([p for _,p in q_dict.items()])
                q = q_deg / 180 * np.pi
                print("{} read position[rad]: {}, position [deg]: {}".format(self.read_pos_cnt, q, q_deg))
                return q
            except BaseException:
                if not failed_once:
                    print("{} read position fail".format(self.read_pos_cnt))
                failed_once = True


    def read_vel(self):
        self.read_vel_cnt += 1
        failed_once = False
        while True:
            try:
                dq_dict = self.robot.bus.sync_read("Present_Velocity")
                dq_deg_p_s = np.array([v for _,v in dq_dict.items()])
                dq = dq_deg_p_s / 180 * np.pi
                print("{} read velocity [rad]: {}".format(self.read_vel_cnt, dq))
                return dq
            except BaseException:
                if not failed_once:
                    print("{} read velocity fail".format(self.read_vel_cnt))
                failed_once = True


    def in_bounds(self, q):
        # just make sure that the robot doesn't go into the table so only check the z
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model,self.data)
        ee_pos = self.data.site("gripper").xpos
        ee_pos_rpz = xyz_to_rpz(ee_pos)
        return ee_pos_rpz[-1] >= self.arm.rpz_low[-1]


    def send_action(self, action):
        print("send_action(), action = {}".format(action))
        np_action = np.array(action) / np.pi * 180
        if not self.robot.is_connected:
            return
        keys = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
        print("send_action(), np_action = {}".format(np_action))
        # the unit of each action element is radian so convert to degree
        self.robot.send_action(dict(zip(keys, np_action)))


    def reset(self,
              seed: int | None = None,
              options: dict | None = None,
              ):
        print("resetting hardware env")
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
        time.sleep(2.0)

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


    def step(self, action_scale):
        # This needs to replicate how the simulation environment works, which
        # first applies/sends the action and then steps the simulation (sleep on hardware).
        action = self.arm.action_scale_to_pos(action_scale,
                                              mj_model=self.model,
                                              qpos=self.read_pos())

        self.send_action(action)
        self.arm.step_sleep(display_rate=True)

        # this just calculates reward, gets observation etc
        return self.arm.step(action, mj_model=self.model, mj_data=self.data)


    def get_goal_xyz(self):
        return rpz_to_xyz(self.arm.goal_rpz)


    def get_ee_pos(self):
        return self.arm.forward_kinematics_ee(
            self.data.qpos, mj_model=self.model, mj_data=self.data
        )


    def __del__(self):
        if hasattr(self,"robot"):
            self.robot.disconnect()
