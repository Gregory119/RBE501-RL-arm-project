import numpy as np
import mujoco
from gymnasium.spaces import Box
from dataclasses import dataclass

import copy


def rpz_to_xyz(rpz):
    rho, phi, z = rpz
    # transform cylindrical to cartesian coordinates
    x = rho*np.cos(phi)
    y = rho*np.sin(phi)
    return np.array([x,y,z])


class Arm:
    """
    Contains common arm environment functionality.
    """

    def __init__(self,
                 get_pos_fn,
                 get_vel_fn,
                 load_env_fn,
                 should_truncate_fn,
                 vis_fn,
                 set_obs_space_fn,
                 np_random,
                 enable_normalize = True,
                 enable_terminate = False,
                 **kwargs):
        """Constructor
        
        :param enable_normalize If True, normalizes the observation
        data, which improves reward performance.
        :param enable_terminate If True, episodes are terminated when the ee
        position is within a radius of the goal. Enabling this reduces
        reward performance because future rewards in terminal states have a reward of
        zero, resulting in the ee avoiding the goal region."""

        self.get_pos_fn = get_pos_fn
        self.get_vel_fn = get_vel_fn
        self.load_env_fn = load_env_fn
        self.should_truncate_fn = should_truncate_fn
        self.vis_fn = vis_fn
        self.set_obs_space_fn = set_obs_space_fn
        self.np_random = np_random
        
        self.enable_normalize = enable_normalize
        self.enable_terminate = enable_terminate
        if self.enable_normalize:
            observation_space = Box(low=-1, high=1, shape=(15,), dtype=np.float64)
        else:
            observation_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64)
        self.set_obs_space_fn(observation_space)

        self.goal_rpz = self.sample_pos_rpz()


    def step(self, action, mj_data):
        obs = self.get_obs()

        ####### Defining reward
        # Using a decaying exponential function based on the distance from the
        # goal for the reward gives a maximum reward when the distance to the
        # goal is zero. The policy is encouraged to move to this state as soon
        # as possible so that the maximum reward is obtained for each step of
        # the environment.
        ee_pos = mj_data.site("gripper").xpos
        dist = np.linalg.norm(ee_pos - rpz_to_xyz(self.goal_rpz))
        assert(dist > 0)

        reward = np.exp(-10*dist)

        # truncation by timeout is set externally
        truncated = self.should_truncate_fn()
        # allow termination in the goal region, if enabled, but reward
        # performance is better when this is disabled
        goal_radius = 0.02
        terminated = self.enable_terminate and dist < goal_radius

        info = {
            "terminated": terminated,
            "truncated": truncated,
        }

        self.vis_fn()
        
        return obs, reward, terminated, truncated, info


    def sample_pos_rpz(self):
        # the robot is facing in the -y direction

        # set limits in cylindrical coordinates
        # rho, phi, z
        rho, phi, z = self.np_random.uniform(
            low = np.array([0.1143,-np.pi,0.075]),# lower bound
            high = np.array([0.4064,0,0.25]), # upper bound
        )

        return np.array([rho, phi, z])


    def reset(self, model, mj_data):
        #Randomization of goal point
        self.goal_rpz = self.sample_pos_rpz()
        self.load_env_fn()
        mujoco.mj_forward(model, mj_data)
        
        return self.get_obs()

    
    def get_obs(self):
        q = self.get_pos_fn()
        dq = self.get_vel_fn()
        
        if self.enable_normalize:
            # normalize observation data
            q_max = np.pi
            q_norm = copy.deepcopy(q) / q_max
            dq_max = np.pi
            dq_norm = copy.deepcopy(dq) / dq_max
            goal_rpz_norm = copy.deepcopy(self.goal_rpz)
            goal_rpz_norm[0] = (goal_rpz_norm[0]-0.1143)/(0.4064-0.1143)
            # remember y range is negative
            goal_rpz_norm[1] = -goal_rpz_norm[1]/np.pi
            goal_rpz_norm[2] = (goal_rpz_norm[2]-0.075)/(0.25-0.075)
            # each normalized goal element is now within [0,1], but the other
            # observations are within [-1,1] so adjust the goal elements to be
            # within [-1,1]
            goal_rpz_norm = goal_rpz_norm*2-1
            obs = np.concatenate([q_norm, dq_norm, goal_rpz_norm]).ravel() #edited to return goal
            return obs
        else:
            return np.concatenate([q, dq, self.goal_rpz]).ravel()
