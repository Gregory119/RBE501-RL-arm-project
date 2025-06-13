#!/usr/bin/env python
import argparse, os, time
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback



#TensorBoard logging callbacks

class TBCallback(BaseCallback):
    def __init__(self, writer, verbose=0):
        super().__init__(verbose)
        self.writer = writer
        self.ep_rew = 0
        self.ep_len = 0
        self.ep_succ = 0

    def _on_step(self) -> bool:
        self.ep_rew += self.locals["rewards"][0]
        self.ep_len += 1
        self.ep_succ += self.locals["infos"][0].get("success", 0)
        return True

    def _on_rollout_end(self):
        self.writer.add_scalar("Rollout/episode_length", self.ep_len, self.num_timesteps)
        self.writer.add_scalar("Rollout/episode_reward", self.ep_rew, self.num_timesteps)
        self.writer.add_scalar("Rollout/episode_success", self.ep_succ, self.num_timesteps)
        self.ep_rew = 0
        self.ep_len = 0
        self.ep_succ = 0


#start arm env
def make_env(rank: int, seed: int = 0):
    def _init():
        import gymnasium_env
        env = gym.make("Arm-v0")
        env = Monitor(env)  #log episode 
        env.reset(seed=seed + rank)
        return env
    return _init


#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--total-steps", type=int, default=50_000)
parser.add_argument("--logdir", type=str, default="runs/sac_arm")
args = parser.parse_args()


#train
if __name__ == "__main__":
    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(args.logdir)

    # parallel envs
    env_fns = [make_env(i) for i in range(8)]
    env = SubprocVecEnv(env_fns)

    # SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=args.logdir,
        device="auto",
    )

    # checkpoint saving , TensorBoard scalar logging
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000 // env.num_envs,
        save_path="checkpoints",
        name_prefix="sac_arm",
    )
    tb_cb = TBCallback(writer)
    callback = CallbackList([checkpoint_cb, tb_cb])

    # Begin training
    print("  Training...")
    model.learn(total_timesteps=args.total_steps, callback=callback)

    # final save
    model.save("checkpoints/sac_lv0.zip")
    env.close()
    writer.close()
    print("Done")
