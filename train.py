#!/usr/bin/env python
import argparse, os, time
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium_env



#TensorBoard logging callbacks

class TBCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rew = 0
        self.ep_len = 0
        self.ep_succ = 0

    def _on_step(self) -> bool:
        self.ep_rew += self.locals["rewards"][0]
        self.ep_len += 1
        term = self.locals["infos"][0].get("terminated", 0)
        ep_end = term or self.locals["infos"][0].get("truncated", 0)
        if ep_end:
            self.logger.record("episode/length", self.ep_len, self.num_timesteps)
            self.logger.record("episode/reward", self.ep_rew, self.num_timesteps)
            self.logger.record("episode/terminated", term, self.num_timesteps)
            self.ep_rew = 0
            self.ep_len = 0
            
        return True


#start arm env
def make_env(rank: int, vis: bool = False, seed: int = 0):
    def _init():
        render_mode=None
        if vis:
            render_mode='human'

        env = gym.make("Arm-v0",render_mode=render_mode)
        env = Monitor(env)  #log episode 
        env.reset(seed=seed + rank)
        return env
    return _init


#train
if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=2000_000)
    parser.add_argument("--logdir", type=str, default="runs/sac_arm")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--vis", help="enable human render mode on the environments", action="store_true")
    parser.add_argument ("--checkpoint", type=str)
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    # parallel envs
    env_fns = [make_env(i,vis=args.vis) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)

    # SAC agent, look for checkpoint argument
    if args.checkpoint and os.path.isfile(args.checkpoint):
        model = SAC.load(
            args.checkpoint,
            env=env,
            tensorboard_log=args.logdir,
            device="auto",
        )
    else: 
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=args.logdir,
            device="auto",
        )

    # checkpoint saving , TensorBoard scalar logging
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // env.num_envs,
        save_path="checkpoints",
        name_prefix="sac_arm",
    )
    tb_cb = TBCallback()
    callback = CallbackList([checkpoint_cb, tb_cb])

    # Begin training
    print("  Training...")
    model.learn(total_timesteps=args.total_steps, callback=callback)

    # final save
    model.save("checkpoints/sac_lv0.zip")
    env.close()
    print("Done")
