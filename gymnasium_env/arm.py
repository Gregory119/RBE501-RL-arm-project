from os import path
import numpy as np
import mujoco 
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


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

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        

        if hasattr(self.model, "site_name2id"):           # mujoco-py style
                    self.ee_sid = self.model.site_name2id("gripper")
        else:                                             # official bindings
                    self.ee_sid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper"
                    )

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

        self.goal   = np.zeros(3, dtype=np.float32) #initialize goal point
        self.goal_radius = 0.02 #how close the ee needs to be to the goal to be considered successful
        self.success = False

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        
        obs = self._get_obs()
        

        ####### Defining reward
        ee_pos = self.data.site_xpos[self.ee_sid]
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
        
        return obs, reward, terminated, False, info



    def reset_model(self):
        
        #Randomization of goal point
        self.goal = self.np_random.uniform(
            low = np.array([0.15, -0.15, 0.05]),  #lower bound
            high = np.array([0.30, 0.15, 0.20]),  #upper bound
        )
        self.success = False
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel, self.goal]).ravel() #edited to return goal
    
