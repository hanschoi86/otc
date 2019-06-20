import numpy as np
import os
import argparse
from obstacle_tower_env import ObstacleTowerEnv, ActionFlattener

from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import PPO2

seed = np.random.randint(0, 100)

def make_env():
    def _init():
        env = ObstacleTowerEnv('./ObstacleTower/obstacletower', worker_id=seed,
                               retro=True, config={'total-floors': 10}, greyscale=True, timeout_wait=600)
        env._flattener = ActionFlattener([2, 3, 2, 1])
        env._action_space = env._flattener.action_space
        return env
    return _init

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
    parser.add_argument('--gamma', help='PPO gamma', type=float, default=0.999)
    parser.add_argument('--num_timesteps', type=int, default=int(10e8))
    parser.add_argument('--learning_rate', type=float, default=.0000625)
    parser.add_argument('--num_evaluation', type=int, default=5)
    args = parser.parse_args()

    # Create vectorized multi-environment to run on 12 cores
    env = SubprocVecEnv([make_env()])

    # Create PPO model for GPU
    multimodel = PPO2(CnnPolicy, env, verbose=1, gamma=args.gamma, learning_rate=args.learning_rate)
    multimodel = multimodel.load('models/ppo/multimodel', env)
    for i in range(args.num_evaluation):
        print("Starting evaluation", i)
        print(run_episode(env, multimodel))
