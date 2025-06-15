import os
import gymnasium as gym
from gymnasium_env.arm import ArmEnv

xml_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../gymnasium_env/SO-ARM100/Simulation/SO101/so101_new_calib.xml"
    )
)

env = ArmEnv(
    xml_file=xml_path,
    render_mode="human"
)

print("EE site ID  :", env.ee_sid)
print("EE position :", env.data.site_xpos[env.ee_sid])

obs, _ = env.reset()
print("EE pos after reset:", env.data.site_xpos[env.ee_sid])
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()