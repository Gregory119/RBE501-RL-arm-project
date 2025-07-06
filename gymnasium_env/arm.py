from os import path
import numpy as np
import mujoco 
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dataclasses import dataclass
import copy

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.04,
}


class ArmEnv(MujocoEnv):
    

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

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
        frame_skip: int = 2,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        max_episode_steps = 500,
        enable_normalize = True,
        enable_terminate = False,
        **kwargs,
    ):
        """Constructor

        :param max_episode_steps Number of steps before timeout/truncation.
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

        self.enable_normalize = enable_normalize
        self.enable_terminate = enable_terminate
        if self.enable_normalize:
            observation_space = Box(low=-1, high=1, shape=(15,), dtype=np.float64)
        else:
            observation_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64)

        self._sample_goal()

        self.load_data = self.LoadData(xml_file=xml_file,
                                       frame_skip=frame_skip,
                                       observation_space=observation_space,
                                       default_camera_config=default_camera_config,
                                       kwargs=kwargs)

        self.max_episode_steps = max_episode_steps
        self.reset_model()
        
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }


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

        
    def step(self, action):
        self.steps += 1
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()

        ####### Defining reward
        # Using a decaying exponential function based on the distance from the
        # goal for the reward gives a maximum reward when the distance to the
        # goal is zero. The policy is encouraged to move to this state as soon
        # as possible so that the maximum reward is obtained for each step of
        # the environment.
        ee_pos = self.data.site("gripper").xpos
        dist = np.linalg.norm(ee_pos - self.goal_to_xyz())
        assert(dist > 0)

        reward = np.exp(-10*dist)

        truncated = self.steps >= self.max_episode_steps
        # allow termination in the goal region, if enabled, but reward
        # performance is better when this is disabled
        goal_radius = 0.02
        terminated = self.enable_terminate and dist < goal_radius

        info = {
            "terminated": terminated,
            "truncated": truncated,
        }

        if self.render_mode == "human":
            self.render()
            # Visualize the site frames. This is a bit of a hack but it works
            # and is simple. This is specifically done after self.render() to
            # ensure that the renderer exists.
            self.mujoco_renderer.viewer.vopt.frame = mujoco.mjtFrame.mjFRAME_SITE
        
        return obs, reward, terminated, truncated, info


    def _sample_goal(self):
        # the robot is facing in the -y direction

        # set limits in cylindrical coordinates
        # rho, phi, z
        rho, phi, z = self.np_random.uniform(
            low = np.array([0.1143,-np.pi,0.075]),# lower bound
            high = np.array([0.4064,0,0.25]), # upper bound
        )

        self.goal_rpz = np.array([rho, phi, z])


    def goal_to_xyz(self):
        rho, phi, z = self.goal_rpz
        # transform cylindrical to cartesian coordinates
        x = rho*np.cos(phi)
        y = rho*np.sin(phi)
        return np.array([x,y,z])


    # override
    def reset_model(self):
        self.steps=0
        #Randomization of goal point
        self._sample_goal()
        self._load_env()
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs()

    
    def _get_obs(self):
        if self.enable_normalize:
            # normalize observation data
            q_max = np.pi
            q_norm = copy.deepcopy(self.data.qpos) / q_max
            dq_max = np.pi
            dq_norm = copy.deepcopy(self.data.qvel) / dq_max
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
            return np.concatenate([self.data.qpos, self.data.qvel, self.goal_rpz]).ravel()
    

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
            pos=self.goal_to_xyz(),
            quat=[0, 1, 0, 0],
        )
        
        # compile model and create data
        model = spec.compile()
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data
