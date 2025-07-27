from gymnasium.envs.registration import register

from .arm_sim import ArmSimEnv

try:
    from .arm_hw import ArmHwEnv
    enable_hw = True
except ImportError:
    enable_hw = False


register(
    id="ArmSim-v0",
    entry_point="gymnasium_env.arm_sim:ArmSimEnv",
    max_episode_steps=300
)

if enable_hw:
    register(
        id="ArmHw-v0",
        entry_point="gymnasium_env.arm_hw:ArmHwEnv",
        max_episode_steps=200
        )
