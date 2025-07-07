from os import path
import numpy as np
import mujoco
import gymnasium as gym
import copy

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dataclasses import dataclass


class ArmHw(gym.Env):

    def __init__(self):
        # todo: throw if robot is None
        # todo: disallow rendering

    def _load_env(self):
        # todo: configure and connect to robot
        # todo: reset hardware position (convert between radians and degrees)


    def pre_step(self):
        # todo: does this need a sleep to ensure a consistent dt between step calls?

        # This does not need a full dynamics simulation step, only a forward
        # kinematics update and visualization. Here just update the cached
        # observation and then set the new state.
        # todo: update cached observation (unnormalized/raw directly from hardware)

        # todo: 
        #self.set_state()

        
    def get_obs():
        # todo: normalize cached raw observation data (pull normalization logic
        # from ArmEnv class)


    def load_env():
        # first load the simulation
        super().load_env()

        # todo: create physical robot interface and reset position


    def should_truncate():
        # todo: check if out of bounds
