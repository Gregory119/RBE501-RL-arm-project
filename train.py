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
        self.first_step = True

    def _on_step(self) -> bool:
        # initialize variables once the number of environments can be detected
        if self.first_step:
            self.num_envs = self.locals["env"].num_envs
            self.ep_reward_sums = np.zeros((self.num_envs))
            self.ep_lens = np.zeros((self.num_envs))
            self.first_step = False

        # rewards exist for each parallel environment so track the accumulated
        # reward for each environment and then log the average of the ones that
        # are done
        dones = self.locals["dones"]
        if np.sum(dones) > 0:
            # average the accumulated rewards for the done episodes
            ep_rew = np.sum(self.ep_reward_sums[dones])/np.sum(dones)
            # average the lengths of the dones episodes
            ep_len = np.sum(self.ep_lens[dones])/np.sum(dones)
            self.logger.record("episode/length", ep_len, self.num_timesteps)
            self.logger.record("episode/reward", ep_rew, self.num_timesteps)

        # The reward just received for the done episodes is actually for the
        # first step of the new episode, which is why they weren't added to the
        # sum/accumulated reward before logging. Next reset the done episode
        # rewards and then add the new set of rewards to the current array of
        # reward sums.

        # set the reward sum to zero for each done episode
        self.ep_reward_sums[dones] = 0
        self.ep_reward_sums += self.locals["rewards"]

        # reset done episode lengths before increment
        self.ep_lens[dones] = 0
        self.ep_lens += 1
            
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
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    # parallel envs
    env_fns = [make_env(i,vis=args.vis) for i in range(args.num_envs)]
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
