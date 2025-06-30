from os import path
import numpy as np
import mujoco 
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dataclasses import dataclass

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
        max_episode_steps=500,
        **kwargs,
    ):
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

        # self.goal = np.zeros(3, dtype=np.float32) #initialize goal point
        self.goal = np.array([0.5,0.1,0.1])
        self.goal_radius = 0.02 #how close the ee needs to be to the goal to be considered successful

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
        self.reached_goal = False


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
        # The problem with using the negative distance as the reward is that the
        # maximum accumulated reward depends on the start configuration. More
        # specifically, the policy will only achieve a zero accumulated reward
        # if the robot can move to the goal in one environment step.

        # Instead the maximum accumulated reward can be set by making each
        # reward equal to the difference in distance from the goal, normalized
        # by the goal distance at the start of the episode. This ratio will be 1
        # when the goal is reached (assuming zero tolerance), so it can be
        # multiplied to give the desired maximum accumulated reward. The only
        # problem is that the accumulated reward can be slightly larger than the
        # desired max due to accumulated floating point resolution error in the
        # dist difference.
        ee_pos = self.data.site("gripper").xpos
        dist = np.linalg.norm(ee_pos - self.goal)
        assert(dist > 0)
        diff_dist = self.prev_dist - dist
        # if distance difference > 0 => moving toward the goal => reward > 0
        # if distance difference < 0 => moving away from the goal => reward < 0
        # if moving away from goal, then double the reward for a higher penalty
        scale = 1 if diff_dist > 0 else 2
        reward = 100*scale*diff_dist/self.start_dist
        self.prev_dist = dist

        # lets never terminate, only truncate so that when the robot gets to the
        # goal region it learns to stay there
        if dist < self.goal_radius and not self.reached_goal:
            reward += 50.0
            self.reached_goal = True

        # if the goal is not reached anymore, then remove the goal reward
        if self.reached_goal and dist > self.goal_radius:
            reward -= 50.0
            self.reached_goal = False

        truncated = self.steps >= self.max_episode_steps
        terminated = truncated and self.reached_goal

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

        # transform cylindrical to cartesian coordinates
        x = rho*np.cos(phi)
        y = rho*np.sin(phi)
        return np.array([x,y,z])

    # override
    def reset_model(self):
        self.steps=0
        self.reached_goal=False
        #Randomization of goal point
        self.goal = self._sample_goal()
        self._load_env()
        mujoco.mj_forward(self.model, self.data)

        # setup initial distance for calculating the reward after the first step
        ee_pos = self.data.site("gripper").xpos
        self.prev_dist = np.linalg.norm(ee_pos - self.goal)
        self.start_dist = self.prev_dist
        assert(self.start_dist > 0)
        
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
