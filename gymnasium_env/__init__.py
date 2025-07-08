from gymnasium.envs.registration import register

from .arm_sim import ArmSimEnv
from .arm_hw import ArmHwEnv

register(
    id="ArmSim-v0",
    entry_point="gymnasium_env.arm_sim:ArmSimEnv",
    max_episode_steps=500
)

register(
    id="ArmHw-v0",
    entry_point="gymnasium_env.arm_hw:ArmHwEnv",
    max_episode_steps=500
)
