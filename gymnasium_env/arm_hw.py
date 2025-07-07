from os import path
import numpy as np
import mujoco
import gymnasium as gym
import copy
import time

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dataclasses import dataclass

from lerobot.common.robots import (
    Robot,
    RobotConfig,
    make_robot_from_config,
    so101_follower,
)


class ArmHw(gym.Env):

    def __init__(self,
                 robot_cfg,
                 xml_file: str | None = None,
                 rate_hz: int = 250,
                 enable_normalize = True,
                 enable_terminate = False,
                 ):
        # overwrite robot configuration to use degrees (radians is not an option)
        robot_cfg.use_degrees = True

        self.robot = make_robot_from_config(robot_cfg)
        self.robot.connect()

        # load the robot model
        xml_file = path.join(
            path.dirname(__file__),
            "gymnasium_env/SO-ARM100/Simulation/SO101/scene.xml"
            )
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(model)

        get_pos_fn = self.read_pos
        get_vel_fn = self.read_vel
        load_env_fn = lambda: self.load_env()
        should_truncate_fn = lambda: not self.in_bounds()
        visualize = lambda: None

        observation_space = None
        def set_obs_space(obs_space):
            nonlocal observation_space
            observation_space = obs_space

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


    def read_pos(self):
        q_dict = self.robot.bus.sync_read("Present_Position")
        q_deg = np.array([p for _,p in q_dict.items()])
        q = q_deg / 180 * np.pi
        return q


    def read_vel(self):
        dq_dict = self.robot.bus.sync_read("Present_Velocity")
        dq_deg_p_s = np.array([v for _,v in q_dict.items()])
        dq = dq_deg_p_s / 180 * np.pi
        return dq


    def in_bounds(self):
        q = self.read_pos()
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model,self.data)
        ee_pos = data.site("gripper").xpos
        return self.arm.in_bounds(ee_pos)


    def _send_action(self, action):
        keys = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
        self.robot.send_action(dict(zip(keys, action)))


    def reset():
        self.load_env()


    def load_env(self):
        # robot should already be connected so just reset the position
        self._send_action([0,0,0,0,0,0])

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
        self.arm.step_sleep(display_rate=True)
        self.robot.send_action(action)


    def __del__(self):
        self.robot.disconnect()
