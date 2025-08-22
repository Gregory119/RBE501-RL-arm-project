import numpy as np
import mujoco
from gymnasium.spaces import Box
from dataclasses import dataclass

import copy
import time


def rpz_to_xyz(rpz):
    rho, phi, z = rpz
    # transform cylindrical to cartesian coordinates
    x = rho*np.cos(phi)
    y = rho*np.sin(phi)
    return np.array([x,y,z])


def xyz_to_rpz(xyz):
    assert(xyz.shape == (3,))
    # convert xyz position to rpz and compare to bounds
    rho = np.linalg.norm(xyz[:2])
    x, y, z = xyz
    phi = np.atan2(y, x)
    pos_rpz = np.array([rho, phi, z])
    return pos_rpz


class Arm:
    """
    Contains common arm environment functionality.
    """

    def __init__(self,
                 rate_hz: int,
                 get_pos_fn,
                 load_env_fn,
                 should_truncate_fn,
                 vis_fn,
                 set_obs_space_fn,
                 np_random,
                 enable_normalize = True,
                 enable_terminate = False,
                 rpz_low = None,
                 rpz_high = None,
                 assert_obs = True,
                 default_goal_rpz = None,
                 **kwargs):
        """Constructor
        
        :param enable_normalize If True, normalizes the observation
        data, which improves reward performance.
        :param enable_terminate If True, episodes are terminated when the ee
        position is within a radius of the goal. Enabling this reduces
        reward performance because future rewards in terminal states have a reward of
        zero, resulting in the ee avoiding the goal region.
        :param default_goal_rpz Set this to a deterministic goal in cylindrical
        coordinates. If None, a random goal will be sampled.

        """

        self.rate_hz = rate_hz
        self.get_pos_fn = get_pos_fn
        self.load_env_fn = load_env_fn
        self.should_truncate_fn = should_truncate_fn
        self.vis_fn = vis_fn
        self.set_obs_space_fn = set_obs_space_fn
        self.np_random = np_random
        
        self.enable_normalize = enable_normalize
        self.enable_terminate = enable_terminate
        if self.enable_normalize:
            observation_space = Box(low=-1, high=1, shape=(9,), dtype=np.float64)
        else:
            observation_space = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64)
        self.set_obs_space_fn(observation_space)

        # workspace bounds
        self.rpz_low = np.array([0.0254*4.5,-np.pi,0.0254*3])
        self.rpz_high = np.array([0.0254*16.25,0,0.0254*10])
        if rpz_low is not None:
            self.rpz_low = np.array(rpz_low)
            assert self.rpz_low.shape == (3,)
        if rpz_high is not None:
            self.rpz_high = np.array(rpz_high)
            assert self.rpz_high.shape == (3,)

        self.assert_obs = assert_obs
        self.default_goal_rpz = default_goal_rpz
        self.goal_rpz = self.sample_pos_rpz()

        self.prev_step_ts_ns = None


    def step_sleep(self, display_rate=False):
        """Call this within step() to sleep the required amount to meet the
        desired step rate. This of course cannot take time away to speed up the
        actual step rate."""
        step_ts_ns = time.perf_counter_ns()
        if self.prev_step_ts_ns is not None:
            dur = (step_ts_ns - self.prev_step_ts_ns)*1e-9
            desired_dur = 1 / self.rate_hz
            dur_diff = desired_dur - dur
            if dur_diff > 0:
                time.sleep(dur_diff)

        # display the measured rate
        if display_rate and self.prev_step_ts_ns is not None:
            step_ts_ns = time.perf_counter_ns()
            dur = (step_ts_ns - self.prev_step_ts_ns)*1e-9
            actual_rate = 1 / dur
            # note that the actual rate cannot go faster the 60 Hz because
            # that's the limit of the mujoco renderer and probably the
            # physical monitor limit
            print("measured rate [Hz]: {}".format(actual_rate))

        # step() is only being performed at this point and a sleep might have
        # occurred in the above logic, so update the previous step timestep
        # accordingly
        self.prev_step_ts_ns = time.perf_counter_ns()


    def action_scale_to_pos(self, action_scale, mj_model, qpos):
        # Clip the action by limiting the desired relative change in joint
        # positions. This must be called before the action is applied

        q_low, q_high = mj_model.jnt_range.T
        assert(np.all(q_high > q_low))

        # clip upper and lower bounds within [-pi,pi] so that the hardware
        # protocol doesn't attempt to send a negative value and fail
        q_low = np.clip(q_low, a_min=np.full(shape=q_low.shape, fill_value=-np.pi), a_max=np.full(shape=q_low.shape, fill_value=np.pi))
        q_high = np.clip(q_high, a_min=np.full(shape=q_high.shape, fill_value=-np.pi), a_max=np.full(shape=q_high.shape, fill_value=np.pi))

        max_rel = (q_high - q_low)*0.05
        return qpos + action_scale*max_rel


    def step(self, action, mj_model, mj_data):
        #print("arm.step(): action = {}".format(action))
        q_low, _ = mj_model.jnt_range.T
        obs = self.get_obs(q_low=q_low)

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

        # # if ee too close to ground, then penalize
        # assert self.goal_rpz[-1] >= self.rpz_low[-1], "goal: {}, rpz_low: {}".format(self.goal, self.rpz_low)
        # if xyz_to_rpz(ee_pos)[-1] < self.rpz_low[-1]:
        #     reward -= 10

        # # If holding the action constant would cause the robot to go into the
        # # ground, then penalize. This should reduce the chance of an action
        # # causing the physical robot to hit the table.
        # ee_pos_hold_action = self.forward_kinematics_ee(qpos=action, mj_model=mj_model, mj_data=mj_data)
        # ee_pos_hold_action_rpz = xyz_to_rpz(ee_pos_hold_action)
        # if ee_pos_hold_action_rpz[-1] < self.rpz_low[-1]:
        #     reward -= 1

        # side note: truncation by timeout is set externally
        truncated = self.should_truncate_fn(q=obs[:6])
        # if truncated:
        #     print("truncated")
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
        if self.default_goal_rpz is None:
            # the robot is facing in the -y direction

            # set limits in cylindrical coordinates
            # rho, phi, z
            rho, phi, z = self.np_random.uniform(
                low = self.rpz_low,
                high = self.rpz_high,
                )

            return np.array([rho, phi, z])

        return self.default_goal_rpz


    def in_bounds(self, pos_xyz):
        pos_rpz = xyz_to_rpz(pos_xyz)

        ret = np.all(pos_rpz < self.rpz_high) and np.all(pos_rpz > self.rpz_low)
        # if not ret:
        #     print("arm out of bounds")
        return ret


    def forward_kinematics_ee(self, qpos, mj_model, mj_data, site_name = "gripper"):
        # store the current state
        q_orig = mj_data.qpos.copy()

        # calculate FK
        mj_data.qpos[:] = qpos
        mujoco.mj_kinematics(mj_model, mj_data)
        ee_pos = mj_data.site("gripper").xpos.copy()

        # restore state
        mj_data.qpos[:] = q_orig
        mujoco.mj_kinematics(mj_model, mj_data)

        return ee_pos


    def reset(self, mj_model, mj_data):
        #Randomization of goal point
        self.goal_rpz = self.sample_pos_rpz()
        self.load_env_fn()
        mujoco.mj_forward(mj_model, mj_data)

        q_low, _ = mj_model.jnt_range.T
        return self.get_obs(q_low=q_low)

    
    def get_obs(self, q_low):
        assert(q_low.shape == (6,))
        q = self.get_pos_fn()
        
        if self.enable_normalize:
            # normalize observation data
            q_new = (copy.deepcopy(q) - q_low) / (2*np.pi) # [0,1]
            q_new = q_new*2-1 # [-1,1]
            if self.assert_obs:
                assert np.all(np.abs(q_new) <= 1.05), "q_new = {}, q = {}, q_low = {}".format(q_new, q, q_low)
            np.clip(q_new, a_min=-1, a_max=1)
            goal_rpz_new = np.array(self.goal_rpz)
            if self.assert_obs:
                assert(goal_rpz_new.shape == (3,))
                assert(self.rpz_low.shape == (3,))
                assert(self.rpz_high.shape == (3,))

            goal_rpz_new = (goal_rpz_new - self.rpz_low)/(self.rpz_high - self.rpz_low)
            # each normalized goal element is now within [0,1], but the other
            # observations are within [-1,1] so adjust the goal elements to be
            # within [-1,1]
            goal_rpz_new = goal_rpz_new*2-1
            if self.assert_obs:
                assert np.all(np.abs(goal_rpz_new) <= 1), "goal_rpz_new = {}".format(goal_rpz_new)

            obs = np.concatenate([q_new, goal_rpz_new]).ravel()
            return obs
        else:
            return np.concatenate([q, dq, self.goal_rpz]).ravel()
