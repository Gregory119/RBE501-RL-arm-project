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


class Arm:
    """
    Contains common arm environment functionality.
    """

    def __init__(self,
                rate_hz: int,
                get_pos_fn,
                get_vel_fn,
                load_env_fn,
                should_truncate_fn,
                vis_fn,
                set_obs_space_fn,
                np_random,
                enable_normalize = True,
                enable_terminate = False,
                mass_scale_range: tuple[float, float] = (1.0, 1.0),
                mass_scale_seed: int | None = None,
                shared_mass_scale: float | None = None,
                mass_scale = None,
                max_episode_steps=500,
                **kwargs):
        """Constructor
        
        :param enable_normalize If True, normalizes the observation
        data, which improves reward performance.
        :param enable_terminate If True, episodes are terminated when the ee
        position is within a radius of the goal. Enabling this reduces
        reward performance because future rewards in terminal states have a reward of
        zero, resulting in the ee avoiding the goal region."""

        """
        :param max_episode_steps Number of steps before timeout/truncation.
        """
        if xml_file is None:
            xml_file = path.join(
                path.dirname(__file__),
                "SO-ARM100", "Simulation", "SO101", "scene.xml"
            )
        if not path.exists(xml_file):
            raise FileNotFoundError(f"Mujoco model not found: {xml_file}")
            
        observation_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64)

        self.goal = np.array([0.5,0.1,0.1])
        self.mass_scale_range = mass_scale_range
        self.mass_scale_seed  = mass_scale_seed
        self.shared_mass_scale = float(shared_mass_scale) if shared_mass_scale is not None else None
        
        self._mass_scale = None 
        self._load_env()
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        self.reset_model()

        self.rate_hz = rate_hz
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

        # workspace bounds
        self.rpz_low = np.array([0.1143,-np.pi,0.075])
        self.rpz_high = np.array([0.4064,0,0.25])

        self.goal_rpz = self.sample_pos_rpz()

        self.prev_step_ts_ns = None


    def _generate_mass_scales(self):
        low, high = self.mass_scale_range
        rng = self.np_random
        self._mass_scales = rng.uniform(low, high, size=self.model.nbody)
        self._mass_scales[0] = 1.0          #fix world body

    def _apply_mass_scales(self, model):
        if self._mass_scale is None:
            if self.shared_mass_scale is not None:
                
                self._mass_scale = self.shared_mass_scale
                print(f" Link mass global scale (shared): {self._mass_scale:.4f}")
            else:
                # scalar
                low, high = self.mass_scale_range
                rng = (
                    np.random.default_rng(self.mass_scale_seed)
                    if self.mass_scale_seed is not None else self.np_random
                )
                self._mass_scale = float(rng.uniform(low, high))
                print(f"Link mass global scale: {self._mass_scale:.4f}")

        # apply the scalar to every arm link
        model.body_mass[1:]    *= self._mass_scale
        model.body_inertia[1:] *= self._mass_scale


    def _load_env(self):
        if hasattr(self,"mujoco_renderer"):
            self.close() # close the renderer
        MujocoEnv.__init__(
            self,
            model_path=self.load_data.xml_file,
            frame_skip=self.load_data.frame_skip,
            observation_space=self.load_data.observation_space,
            default_camera_config=self.load_data.default_camera_config,
            **self.load_data.kwargs,
        )
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


    def step(self, action, mj_model, mj_data):
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
            low = self.rpz_low,
            high = self.rpz_high,
        )

        return np.array([rho, phi, z])


    def in_bounds(self, pos_xyz):
        assert(pos_xyz.shape == (3,))
        # convert xyz position to rpz and compare to bounds
        rho = np.linalg.norm(pos_xyz[:2])
        x, y, z = pos_xyz
        phi = np.atan2(y, x)
        pos_rpz = np.array([rho, phi, z])

        return np.all(pos_rpz < self.rpz_high) and np.all(pos_rpz > self.rpz_low)


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
        dq = self.get_vel_fn()
        
        # add a site for the goal
        spec.worldbody.add_site(
            pos=self.goal,
            quat=[0, 1, 0, 0],
        )
        
        # compile model and create data
        model = spec.compile()
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        self._apply_mass_scales(model)
        data = mujoco.MjData(model)
        return model, data
        if self.enable_normalize:
            # normalize observation data
            q_new = (copy.deepcopy(q) - q_low) / (2*np.pi) # [0,1]
            q_new = q_new*2-1 # [-1,1]
            assert np.all(np.abs(q_new) <= 1.05), "q_new = {}, q = {}, q_low = {}".format(q_new, q, q_low)
            np.clip(q_new, a_min=-1, a_max=1)

            dq_max = 2*np.pi
            dq_new = copy.deepcopy(dq) / dq_max
            assert np.all(np.abs(dq_new) <= 1), "dq_new = {}".format(dq_new)

            assert(self.goal_rpz.shape == (3,))
            assert(self.rpz_low.shape == (3,))
            assert(self.rpz_high.shape == (3,))

            goal_rpz_new = self.goal_rpz.copy()
            goal_rpz_new = (goal_rpz_new - self.rpz_low)/(self.rpz_high - self.rpz_low)
            # each normalized goal element is now within [0,1], but the other
            # observations are within [-1,1] so adjust the goal elements to be
            # within [-1,1]
            goal_rpz_new = goal_rpz_new*2-1
            assert np.all(np.abs(goal_rpz_new) <= 1), "goal_rpz_new = {}".format(goal_rpz_new)

            obs = np.concatenate([q_new, dq_new, goal_rpz_new]).ravel() #edited to return goal
            return obs
        else:
            return np.concatenate([q, dq, self.goal_rpz]).ravel()
