from gymnasium.envs.registration import register

from .arm import ArmEnv

register(
    id="Arm-v0",
    entry_point="gymnasium_env.arm:ArmEnv",
    max_episode_steps = 500
)
