from gymnasium.envs.registration import register

register(
    id="Arm-v0",
    entry_point="gymnasium_env.envs:ArmEnv",
)
