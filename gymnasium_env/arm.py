from os import path
import numpy as np
import mujoco 
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dataclasses import dataclass

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.04,
}


class ArmEnv(MujocoEnv, utils.EzPickle):
    

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
        reset_noise_scale: float = 0.01,
        **kwargs,
    ):
        if xml_file is None:
            xml_file = path.join(
                path.dirname(__file__),
                "SO-ARM100", "Simulation", "SO101", "scene.xml"
            )
        if not path.exists(xml_file):
            raise FileNotFoundError(f"Mujoco model not found: {xml_file}")
            
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64)

        self._reset_noise_scale = reset_noise_scale

        # self.goal = np.zeros(3, dtype=np.float32) #initialize goal point
        self.goal = np.array([0.5,0.1,0.1])
        self.goal_radius = 0.02 #how close the ee needs to be to the goal to be considered successful

        self.load_data = self.LoadData(xml_file=xml_file,
                                       frame_skip=frame_skip,
                                       observation_space=observation_space,
                                       default_camera_config=default_camera_config,
                                       kwargs=kwargs)
        self._load_env()
        
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }

        self.success = False

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
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()

        ####### Defining reward
        ee_pos = self.model.site("gripper").pos
        dist     = np.linalg.norm(ee_pos - self.goal)
        reward = -dist

        self.success = dist < self.goal_radius
        if self.success:
            reward += 5.0 

        #terminate episode
        terminated = False

        info = {
            "reward_distance": -dist,
            "success": self.success

        }

        if self.render_mode == "human":
            self.render()
            # Visualize the site frames. This is a bit of a hack but it works and is simple.
            self.mujoco_renderer.viewer.vopt.frame = mujoco.mjtFrame.mjFRAME_SITE
        
        return obs, reward, terminated, False, info


    def _sample_goal(self):
        # the robot is facing in the -y direction
        return self.np_random.uniform(
            low = np.array([-0.2, -0.25, 0.075]),  #lower bound
            high = np.array([0.2, -0.13, 0.25]),  #upper bound
        )

    def reset_model(self):
        #Randomization of goal point
        self.goal = self._sample_goal()
        print(self.goal)
        self.success = False
        self._load_env()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel, self.goal]).ravel() #edited to return goal
    

    def _initialize_simulation(self) -> tuple["mujoco.MjModel", "mujoco.MjData"]:
        """
        Initialize MuJoCo simulation data structures `mjModel` and `mjData`.
        """
        # load the model specification
        spec = mujoco.MjSpec.from_file(self.fullpath)

        # add sites whose frames will be displayed by default

        # readd the gripper site (not sure why the already existing site is not
        # displayed)
        gripper_body = spec.body("gripper")
        gripper_site = spec.site("gripper")
        gripper_body.add_site(pos=gripper_site.pos,
                              quat=gripper_site.quat)
        
        # add a site for the goal
        spec.worldbody.add_site(
            pos=self.goal,
            quat=[0, 1, 0, 0],
        )
        
        # compile model and create data
        model = spec.compile()
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data
