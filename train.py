#!/usr/bin/env python
import argparse, os, time
import gymnasium as gym
import numpy as np
from pathlib import Path
import os
from typing import List
import re
import signal

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

from torch.utils.tensorboard import SummaryWriter
import gymnasium_env


class LogHelper:
    def __init__(self, logger):
        self.first_step = True
        self.logger = logger

    def on_step(self, num_envs, dones: List[bool], rewards, num_timesteps):
        # initialize variables once the number of environments can be detected
        if self.first_step:
            self.num_envs = num_envs
            self.ep_reward_sums = np.zeros((self.num_envs))
            self.ep_lens = np.zeros((self.num_envs))
            self.first_step = False

        # rewards exist for each parallel environment so track the accumulated
        # reward for each environment and then log the average of the ones that
        # are done
        if np.sum(dones) > 0:
            # average the accumulated rewards for the done episodes
            ep_rew = np.sum(self.ep_reward_sums[dones])/np.sum(dones)
            # average the lengths of the dones episodes
            ep_len = np.sum(self.ep_lens[dones])/np.sum(dones)
            self.logger.record("episode/length", ep_len, num_timesteps)
            self.logger.record("episode/reward", ep_rew, num_timesteps)

        # The reward just received for the done episodes is actually for the
        # first step of the new episode, which is why they weren't added to the
        # sum/accumulated reward before logging. Next reset the done episode
        # rewards and then add the new set of rewards to the current array of
        # reward sums.

        # set the reward sum to zero for each done episode
        self.ep_reward_sums[dones] = 0
        self.ep_reward_sums += rewards

        # reset done episode lengths before increment
        self.ep_lens[dones] = 0
        self.ep_lens += 1


#TensorBoard logging callbacks

class TBCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if not hasattr(self,"log_helper"):
            self.log_helper = LogHelper(self.logger)
        self.log_helper.on_step(num_envs=self.locals["env"].num_envs, dones=self.locals["dones"], rewards=self.locals["rewards"], num_timesteps=self.num_timesteps)
        return True


def sig_handler(sig, frame):
    if sig == signal.SIGINT: # Ctrl-C
        handle_fault()


def handle_fault():
    # Move the robot to a safe position. The action must be in radians.
    if g_hw_env is not None:
        action = np.array([0,-80,90,0,0,0])/180*np.pi
        g_hw_env.send_action(action)


def make_sim_env(rank: int, vis: bool = False, seed: int = 0):
    def _init():
        render_mode=None
        if vis:
            render_mode='human'

        env = gym.make("ArmSim-v0",render_mode=render_mode)
        env.reset(seed=seed + rank)
        return env
    return _init


def make_hw_env():
    # this should only be called once
    env = gym.make("ArmHw-v0")
    env.reset()
    global g_hw_env
    g_hw_env = env
    return env


def main(args):
    os.makedirs(args.logdir, exist_ok=True)

    # make environments
    num_envs = 1
    if hasattr(args,"num_envs") and not args.hw:
        # parallel simulation environment(s)
        num_envs = args.num_envs
    if args.hw:
        # one hardware environment
        env_fns = [make_hw_env]
    else:
        env_fns = [make_sim_env(i,vis=args.vis) for i in range(num_envs)]
    venv = SubprocVecEnv(env_fns)

    device = "auto"
    if args.alg == "SAC":
        alg = SAC
    elif args.alg == "PPO":
        alg = PPO
        device = "cpu"

    # use a regular expression to extract the run number from the existing folder name
    run_num_re = re.compile(r".*run_([0-9]+)")
        
    # Create a logger for training and evaluation. It's possible to use a
    # default logger for training, but not for evaluation.
    folder_start = alg.__name__ + '_'
    log_dir_base = Path(os.path.dirname(__file__), args.logdir, args.mode)
    run_nums = [int(run_num_re.match(folder.name).group(1)) for folder in log_dir_base.iterdir() if str(folder.name).startswith(folder_start) and run_num_re.match(folder.name) is not None]
    run_nums.append(0)
    run_num = max(run_nums)+1
    # if evaluating a model its helpful to know the model training run number
    model = "model_" + str(args.model_num) + "_" if args.mode == "eval" else ""
    log_dir = log_dir_base / (folder_start + model + "run_" + str(run_num))
    logger = configure(str(log_dir), ["tensorboard"])
    
    # SAC agent
    model = alg(
        "MlpPolicy",
        venv,
        verbose=0,
        device=device,
        )
    model.set_logger(logger)

    tb_cb = TBCallback()
    if args.mode=="train":
        # the run number here is one more than the last run (computed automatically)
        model_prefix = folder_start+"run_"+str(run_num)
        final_model_path = "checkpoints/"+model_prefix+".zip"
        # checkpoint saving , TensorBoard scalar logging
        checkpoint_cb = CheckpointCallback(
            save_freq=args.total_steps / args.num_checkpoints // venv.num_envs,
            save_path="checkpoints",
            name_prefix=model_prefix,
            )
        callback = CallbackList([checkpoint_cb, tb_cb])

        # Begin training
        print("Training...")
        model.learn(total_timesteps=args.total_steps, callback=callback)

        # final save
        model.save(final_model_path)
        print("Training Done")
    elif args.mode=="eval":
        print("Evaluating")
        # the model number given on the command line argument corresponds to
        # the training run number the model should be loaded from
        model_prefix = folder_start+"run_"+str(args.model_num)
        final_model_path = "checkpoints/"+model_prefix+".zip"

        # this is replicating the case of evaluating on hardware as close as
        # possible but still using simulation, so there should only be one
        # environment
        
        model = alg.load(final_model_path,print_system_info=True)
        model.set_logger(logger)
        
        log_helper = LogHelper(model.logger)
        
        obs = venv.reset()
        for i in range(args.total_steps):
            action, state = model.predict(obs,deterministic=True)  # agent policy that uses the observation and info
            obs, rewards, dones, info = venv.step(action)
            # vectorized environments are automatically reset at the end of
            # each episode, so no need to do that here

            log_helper.on_step(num_envs=venv.num_envs, dones=dones, rewards=rewards, num_timesteps=model.num_timesteps)
            model.logger.dump(i+1)

        print("Evaluation Done")
    venv.close()


g_hw_env = None

#train
if __name__ == "__main__":
    signal.signal(signal.SIGINT, sig_handler)

    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=4000_000)
    parser.add_argument("--num-checkpoints", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="logs/")
    parser.add_argument("--vis", help="enable human render mode on the environments", action="store_true")
    parser.add_argument("--alg", type=str, choices=["PPO","SAC"], default="SAC")
    parser.add_argument("--hw", help="use hardware environment", action="store_true")

    subparsers = parser.add_subparsers(dest="mode")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--num-envs", type=int, default=8)
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--model-num", type=int, required=True, help="the training run number of the model to load")

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        handle_fault()
        raise
