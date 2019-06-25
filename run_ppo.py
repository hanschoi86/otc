import numpy as np
import os
import argparse
from obstacle_tower_env import ObstacleTowerEnv, ActionFlattener

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import PPO2

def run_episode(env, model):
    done = False
    episode_reward = 0.0
    obs = env.reset()

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    return episode_reward

def run_evaluation(env, model):
    while not env.done_grading():
        run_episode(env, model)
        env.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_filename', default='./ObstacleTower/obstacletower', nargs='?')
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    args = parser.parse_args()

    env = ObstacleTowerEnv(args.environment_filename, docker_training=args.docker_training, 
                           retro=True, greyscale=True, timeout_wait=600)
    env._flattener = ActionFlattener([2, 3, 2, 1])
    env._action_space = env._flattener.action_space
    
    multienv = SubprocVecEnv([lambda: env])
    multimodel = PPO2(CnnPolicy, multienv, verbose=1, gamma=.999, learning_rate=.0000625)
    multimodel = multimodel.load('models/ppo/multimodel5', multienv)

    if env.is_grading():
        episode_reward = run_evaluation(env, multimodel)
    else:
        while True:
            episode_reward = run_episode(env, multimodel)
            print("Episode reward: " + str(episode_reward))
            env.reset()

    env.close()
