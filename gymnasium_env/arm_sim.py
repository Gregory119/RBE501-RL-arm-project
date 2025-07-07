from os import path
import numpy as np
import mujoco 
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dataclasses import dataclass
import copy
import time

from .arm import Arm


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 1.5,
}


class ArmSimEnv(MujocoEnv):

    metadata = {
        "render_modes": [
        "human",
        ],}

    class LoadData:
        def __init__(self,
                     xml_file,
                     frame_skip,
                     observation_space,
                     default_camera_config,
                     kwargs):
            self.xml_file=xml_file
            self.frame_skip=frame_skip
            self.observation_space=observation_space
            self.default_camera_config=default_camera_config
            self.kwargs = kwargs
        

    def __init__(
        self,
        xml_file: str | None = None,
        rate_hz: int = 250,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        enable_normalize = True,
        enable_terminate = False,
        **kwargs,
    ):
        """Constructor

        :param enable_normalize If True, normalizes the observation
        data, which improves reward performance.
        :param enable_terminate If True, episodes are terminated when the ee
        position is within a radius of the goal. Enabling this reduces
        reward performance because future rewards in terminal states have a reward of
        zero, resulting in the ee avoiding the goal region.

        """
        if xml_file is None:
            xml_file = path.join(
                path.dirname(__file__),
                "SO-ARM100", "Simulation", "SO101", "scene.xml"
            )
        if not path.exists(xml_file):
            raise FileNotFoundError(f"Mujoco model not found: {xml_file}")

        # construct the arm class that has common arm environment functionality
        
        get_pos_fn = lambda: self.data.qpos
        get_vel_fn = lambda: self.data.qvel
        load_env_fn = lambda: self.load_env()
        should_truncate_fn = lambda: False
        def visualize():
            if self.render_mode == "human":
                self.render()
                # Visualize the site frames. This is a bit of a hack but it works
                # and is simple. This is specifically done after self.render() to
                # ensure that the renderer exists.
                self.mujoco_renderer.viewer.vopt.frame = mujoco.mjtFrame.mjFRAME_SITE

        observation_space = None
        def set_obs_space(obs_space):
            nonlocal observation_space
            observation_space = obs_space
            
        self.arm = Arm(get_pos_fn=get_pos_fn,
                       get_vel_fn=get_vel_fn,
                       load_env_fn=load_env_fn,
                       should_truncate_fn=should_truncate_fn,
                       vis_fn=visualize,
                       set_obs_space_fn=set_obs_space,
                       np_random=self.np_random,
                       enable_normalize=enable_normalize,
                       enable_terminate=enable_terminate)

        # The number of skip frames specifies how many mujoco timesteps to
        # simulate per call to step(). step() should be called at a simulation
        # interval that matches the desired control/action rate, so calculate
        # the number of skip frames to achieve this.
        self.mj_timestep = 0.001
        self.rate_hz = rate_hz
        mj_rate_hz = 1 / self.mj_timestep
        frame_skip = int(mj_rate_hz / rate_hz)

        self.load_data = self.LoadData(xml_file=xml_file,
                                       frame_skip=frame_skip,
                                       observation_space=observation_space,
                                       default_camera_config=default_camera_config,
                                       kwargs=kwargs)

        self.load_env()

        self.prev_step_ts_ns = None


    def load_env(self):
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


    def step(self, action):
        # If rendering, ensure that step is not called faster than the desired
        # rate. This is not needed when not rendering because the simulation is
        # stepped by the appropriate number of skip frames, so step() should be
        # called as fast as possible.
        if self.render_mode == "human":
            step_ts_ns = time.perf_counter_ns()
            if self.prev_step_ts_ns is not None:
                dur = (step_ts_ns - self.prev_step_ts_ns)*1e-9
                desired_dur = 1 / self.rate_hz
                dur_diff = desired_dur - dur
                if dur_diff > 0:
                    time.sleep(dur_diff)

            # display the measured rate
            if self.prev_step_ts_ns is not None:
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

        self.do_simulation(action, self.frame_skip)
        
        return self.arm.step(action, self.data)


    # override
    def reset_model(self):
        return self.arm.reset(self.model, self.data)


    # override
    def _initialize_simulation(self) -> tuple["mujoco.MjModel", "mujoco.MjData"]:
        """
        Initialize MuJoCo simulation data structures `mjModel` and `mjData`.
        """
        # load the model specification
        spec = mujoco.MjSpec.from_file(self.fullpath)

        # add sites whose frames will be displayed by default

        # re-add the gripper site (not sure why the already existing site is not
        # displayed)
        gripper_body = spec.body("gripper")
        gripper_site = spec.site("gripper")
        gripper_body.add_site(pos=gripper_site.pos,
                              quat=gripper_site.quat)
        
        # add a site for the goal
        spec.worldbody.add_site(
            pos=self.arm.goal_to_xyz(),
            quat=[0, 1, 0, 0],
        )
        
        # compile model and create data
        model = spec.compile()
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)

        model.opt.timestep = self.mj_timestep

        return model, data
