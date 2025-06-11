import gymnasium_env                    # Arm-v0
import gymnasium as gym
import mujoco.viewer




env = gym.make("Arm-v0", render_mode=None)
model, data = env.unwrapped.model, env.unwrapped.data
viewer = mujoco.viewer.launch_passive(model, data)

while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
env.close()
