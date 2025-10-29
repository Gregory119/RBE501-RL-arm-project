from os import path
import numpy as np
import mujoco 
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dataclasses import dataclass
import copy

from scipy.optimize import least_squares

from .arm import Arm, rpz_to_xyz


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
        rate_hz: int = 50,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        enable_normalize = True,
        enable_terminate = False,
        mass_and_inertia_scale = 1.0,
        enable_rand_ee_start_and_goal = True,
        **kwargs,
        ):
        """Constructor

        :param enable_normalize If True, normalizes the observation
        data, which improves reward performance.
        :param enable_terminate If True, episodes are terminated when the ee
        position is within a radius of the goal. Enabling this reduces
        reward performance because future rewards in terminal states have a reward of
        zero, resulting in the ee avoiding the goal region.
        :param enable_rand_ee_start_and_goal If True, the end-effector start
        position and the goal are randomly selected for each episode.

        """
        if xml_file is None:
            xml_file = path.join(
                path.dirname(__file__), "scene.xml"
            )
        if not path.exists(xml_file):
            raise FileNotFoundError(f"Mujoco model not found: {xml_file}")

        # construct the arm class that has common arm environment functionality

        get_qpos_fn = lambda: self.data.qpos
        get_qvel_fn = lambda: self.data.qvel
        load_env_fn = lambda: self.load_env()
        should_truncate_fn = lambda q: False
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


        default_goal_rpz = None
        if not enable_rand_ee_start_and_goal:
            rho = 0.0254*16
            phi = -np.pi/2/4
            z = 0.0254*7
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
                       default_goal_rpz=default_goal_rpz)

        # The number of skip frames specifies how many mujoco timesteps to
        # simulate per call to step(). step() should be called at a simulation
        # interval that matches the desired control/action rate, so calculate
        # the number of skip frames to achieve this.
        self.mj_timestep = 0.001
        mj_rate_hz = 1 / self.mj_timestep
        frame_skip = int(mj_rate_hz / rate_hz)

        self.load_data = self.LoadData(xml_file=xml_file,
                                       frame_skip=frame_skip,
                                       observation_space=observation_space,
                                       default_camera_config=default_camera_config,
                                       kwargs=kwargs)

        self._mass_and_inertia_scale = mass_and_inertia_scale
        self._enable_rand_ee_start_and_goal = enable_rand_ee_start_and_goal
        self.load_env()

        # overwrite action space to use relative position scale
        self.action_space = Box(low=-1, high=1, shape=(6,), dtype=np.float64)


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

        # setting the arm state must be done after loading the environment,
        # otherwise it will have no effect
        if self._enable_rand_ee_start_and_goal:
            self._set_rand_arm_state()

    def get_ee_pos(self):
        return self.arm.forward_kinematics_ee(
            self.data.qpos, mj_model=self.model, mj_data=self.data
        )

    def get_goal_xyz(self):
        return rpz_to_xyz(self.arm.goal_rpz)

    def step(self, action_scale):
        action = self.arm.action_scale_to_pos(action_scale, mj_model=self.model, qpos=self.data.qpos)

        # If rendering, ensure that step is not called faster than the desired
        # rate. This is not needed when not rendering because the simulation is
        # stepped by the appropriate number of skip frames, so step() should be
        # called as fast as possible.
        if self.render_mode == "human":
            self.arm.step_sleep(display_rate=True)
        self.do_simulation(action, self.frame_skip)
        
        return self.arm.step(action, mj_model=self.model, mj_data=self.data)


    # override
    def reset_model(self):
        return self.arm.reset(mj_model=self.model, mj_data=self.data)


    def inverse_kinematics(self, target_xyz, q_init= None, max_iter= 200):
        if q_init is None:
            q_init = self.init_qpos.copy()

        # get the joint limits defined in the xml
        lower, upper = self.model.jnt_range.T

        # nested error function for the solver
        def error(q):
            return self.arm.forward_kinematics_ee(q, mj_model=self.model, mj_data=self.data) - target_xyz

        # numerical IK solver
        sol = least_squares(
            error,
            q_init,
            bounds = (lower,upper),
            xtol = 1e-4, #minimum error for solution
            max_nfev = max_iter, #maximum number of solver steps
        )

        return sol.x if sol.success else None


    def _set_rand_arm_state(self):
        #Sample random point in world space for end effector
        ee_start_xyz = rpz_to_xyz(self.arm.sample_pos_rpz())

        #get start pose for random ee position using inverse kinematics
        qpos = self.inverse_kinematics(ee_start_xyz)

        #If IK solver returns nothing, use added joint noise to randomize
        if qpos is None:
            print("Failed to solve IK")
            noise_pos = self.np_random.uniform(-0.05, 0.05, size=self.model.nq)
            qpos = self.init_qpos + noise_pos

        # randomize initial joint velocity over range approximately observed on
        # hardware
        qvel = self.arm.np_random.uniform(
            low = -15.0,
            high = 15.0,
            size = (6,),
        )
        self.set_state(qpos, qvel)


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
            pos=rpz_to_xyz(self.arm.goal_rpz),
            quat=[0, 1, 0, 0],
        )

        # compile model and create data
        model = spec.compile()
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        self.scale_masses_and_inertias(model)
        data = mujoco.MjData(model)

        model.opt.timestep = self.mj_timestep

        return model, data


    def scale_masses_and_inertias(self, model):
        # apply the scale to every arm link
        model.body_mass[1:]    *= self._mass_and_inertia_scale
        model.body_inertia[1:] *= self._mass_and_inertia_scale
