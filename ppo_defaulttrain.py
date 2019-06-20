# Import libraries
import datetime
import numpy as np
import os
import argparse
from obstacle_tower_env import ObstacleTowerEnv, ActionFlattener

from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import PPO2
import gym


# Create Obstacle Tower environment
def make_env(log_dir, cpu):
    sub_dir = "{}cpu_{}/".format(log_dir, cpu)
    os.makedirs(sub_dir, exist_ok=True)
    def _init():
        env = ObstacleTowerEnv('./ObstacleTower/obstacletower', worker_id=500+cpu,
                               retro=True, config={'total-floors':10}, greyscale=True, timeout_wait=600)
        env._flattener = ActionFlattener([2, 3, 2, 1])
        env._action_space = env._flattener.action_space
        env = Monitor(env, sub_dir)
        return env
    return _init

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_cpu', help='number of cpu cores', type=int, default=32)
    parser.add_argument('--gamma', help='PPO gamma', type=float, default=0.999)
    parser.add_argument('--num_timesteps', type=int, default=int(10e7))
    parser.add_argument('--learning_rate', type=float, default=.0000625)
    args = parser.parse_args()

    num_cpu = args.num_cpu
    log_dir = "models/ppo/"
    os.makedirs(log_dir, exist_ok=True)
    # Create vectorized multi-environment to run on 12 cores
    multienv = SubprocVecEnv([make_env(log_dir, cpu) for cpu in range(num_cpu)])

    # Create PPO model for GPU
    multimodel = PPO2(CnnPolicy, multienv, verbose=1, gamma=args.gamma, learning_rate=args.learning_rate)
    multimodel.learn(total_timesteps=args.num_timesteps)
    multimodel.save('models/ppo/multimodel')
