#!/usr/bin/env python

import argparse, time
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
import gymnasium_env


def make_vis_env(seed: int = 0):
    #Return ArmEnv
    env = gym.make("Arm-v0", render_mode="human")
    env = Monitor(env)
    env.reset(seed=seed)
    
    return DummyVecEnv([lambda: env])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/sac_arm_500000_steps.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    # build visualisation env and load the policy
    vec_env  = make_vis_env()
    model: SAC = SAC.load(args.model, env=vec_env, device="auto")

    
    for ep in range(args.episodes):
        obs = vec_env.reset()
        done, truncated = False, False
        ep_return, ep_len = 0.0, 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            
            ep_return += reward[0]
            ep_len    += 1

            if args.realtime:
                base_dt    = vec_env.envs[0].unwrapped.dt
                frame_skip = vec_env.envs[0].unwrapped.frame_skip
                time.sleep(base_dt * frame_skip)
    vec_env.close()