# -*- coding: utf-8 -*-
"""
Created on Sun May 12 00:40:51 2019

@author: hanschoi
"""

# Import libraries
import datetime
import numpy as np
import glob
import os
import re
import argparse

# Import environment libraries
from otc_prep import Preprocessing
from obstacle_tower_env import ObstacleTowerEnv, ActionFlattener

# Import stable baseline libraries
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, LstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines import PPO2
from stable_baselines.common.atari_wrappers import FrameStack


# Create Obstacle Tower environment


def make_env(log_dir, cpu, env_config):
    sub_dir = "{}/cpu/cpu_{}/".format(log_dir, cpu)
    os.makedirs(sub_dir, exist_ok=True)

    def _init():
        env = ObstacleTowerEnv('./ObstacleTower/obstacletower', worker_id=args.starting_workerid + cpu,
                               retro=True, config=env_config, greyscale=args.greyscale, timeout_wait=600)
        env = Preprocessing(env, reshape_action=args.reshape_action)
        if args.stack_size > 1:
            env = FrameStack(env, args.stack_size)
        env = Monitor(env, sub_dir, allow_early_resets=True)
        return env

    return _init


# Create Evaluation Environment
def make_evalenv(env_config):
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', worker_id=args.starting_workerid + 99,
                           retro=True, config=env_config, greyscale=args.greyscale, timeout_wait=600)
    env = Preprocessing(env, reshape_action=args.reshape_action)
    if args.stack_size > 1:
        env = FrameStack(env, args.stack_size)
    return env


# Run episode for evaluation
def run_episode(env, model):
    done = False
    episode_reward = 0.0
    obs = env.reset()

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    return episode_reward


# Run episode for evaluation, for multi-environment
def run_multiepisode(env, model):
    done = [False, False]
    episode_reward = 0.0
    obs = env.reset()

    while not done[0]:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
    return episode_reward


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--enable_obsnorm', help='enable observation normalization', type=bool, default=False)
    parser.add_argument('--obsnorm_mode', help='observation normalization mode', type=str, default="pixel")
    parser.add_argument('--obsnorm_period', help='observation normalization period', type=int, default=10000)
    parser.add_argument('--num_cpu', help='number of cpu cores', type=int, default=32)
    parser.add_argument('--gamma', help='PPO gamma', type=float, default=0.99)
    parser.add_argument('--num_iterations', type=int, default=50)
    parser.add_argument('--num_timesteps', type=int, default=2500000)
    parser.add_argument('--learning_rate', type=float, default=.00025)
    parser.add_argument('--starting_workerid', type=int, default=np.random.randint(100, 400))
    parser.add_argument('--model_save_dir', type=str)# , default="models/ppo/")
    parser.add_argument('--model_filename', type=str, default="ppo_model")
    parser.add_argument('--verbose_learning', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--policy', type=str, default='Cnn')
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--noptepochs', type=int, default=4)
    parser.add_argument('--nminibatches', type=int, default=4)
    parser.add_argument('--reshape_action', type=bool, default=True)
    parser.add_argument('--greyscale', type=bool, default=True)
    parser.add_argument('--stack_size', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=32)

    parser.add_argument('--starting_floor', type=int, default=0)
    parser.add_argument('--total_floors', type=int, default=10)
    parser.add_argument('--fix_theme', type=bool, default=False)
    parser.add_argument('--default_theme', type=float, default=0.0)
    parser.add_argument('--tower_seed', type=float, default=-1.0)
    args = parser.parse_args()

    default_config = {'allowed-rooms': 2.0,
                  'dense-reward': 1.0,
                  'starting-floor': args.starting_floor,
                  'total-floors': args.total_floors,
                  'agent-perspective': 1.0,
                  'tower-seed': args.tower_seed,
                  'default-theme': args.default_theme,
                  'allowed-floors': 2.0,
                  'allowed-modules': 2.0,
                  'lighting-type': 1.0,
                  'visual-theme': 1.0}

    if args.fix_theme:
        default_config['visual-theme'] = 0.0

    # Create directory for model checkpoints
    os.makedirs(args.model_save_dir, exist_ok=True)
    if args.model_save_dir[-1] != '/':
        args.model_save_dir = args.model_save_dir + '/'
    file_prefix = args.model_save_dir + args.model_filename

    num_cpu = args.num_cpu
    log_dir = args.model_save_dir + 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized multi-environment to run on multiple cores
    multienv = SubprocVecEnv([make_env(log_dir, cpu, default_config) for cpu in range(num_cpu)])

    # Create evaluation environment
    evalenv = make_evalenv(default_config)

    # Set policy
    if args.policy == 'CnnLstm':
        Policy = CnnLstmPolicy
    elif args.policy == 'Cnn':
        Policy = CnnPolicy
    elif args.policy == 'Mlp':
        Policy = MlpPolicy
    elif args.policy == 'CnnLnLstm':
        Policy = CnnLnLstmPolicy
    elif args.policy == 'Lstm':
        Policy = LstmPolicy
    else:
        raise ValueError("Error: set correct policy")

    # Create model
    multimodel = PPO2(Policy, multienv, verbose=args.verbose_learning, gamma=args.gamma,
                      learning_rate=args.learning_rate, n_steps=args.n_steps,
                      tensorboard_log=log_dir, noptepochs=args.noptepochs, nminibatches=args.nminibatches)

    # Load checkpoint
    if len(glob.glob(file_prefix + '_*.pkl')) > 0:
        checkpoint_list = [int(re.search(r"\d+(\.\d+)?", x)[0]) for x in glob.glob(file_prefix + '_*.pkl')]
        checkpoint_list.sort()
        last_checkpoint = checkpoint_list[-1]
        print("Loading checkpoint {}".format(last_checkpoint))
        multimodel = multimodel.load(file_prefix + '_{}.pkl'.format(last_checkpoint), env=multienv)
        print("Checkpoint loaded")
    else:
        last_checkpoint = 0

    # Learn for multiple iterations and evaluate for 5 episodes per iteration
    for iteration in range(last_checkpoint, args.num_iterations):
        print("Beginning Iteration", iteration)
        print("Current Time:", datetime.datetime.now())
        multimodel.learn(total_timesteps=args.num_timesteps, log_interval=args.log_interval, reset_num_timesteps=False)
        multimodel.save(file_prefix + '_{}'.format(last_checkpoint + 1))
        last_checkpoint += 1

        if args.evaluate:
            eval_reward = 0.0
            eval_iteration = 10
            for evaliter in range(eval_iteration):
                if args.policy == 'Cnn':
                    eval_reward += run_episode(evalenv, multimodel)
                else:
                    eval_reward += run_multiepisode(multienv, multimodel)
            eval_reward /= eval_iteration
            print("Iteration {} Evaluation Result: {}".format(iteration, eval_reward))
