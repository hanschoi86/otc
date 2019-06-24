import numpy as np
import argparse
from obstacle_tower_env import ObstacleTowerEnv, ActionFlattener

import torch
from torch.distributions.categorical import Categorical

from rnd.model import CnnActorCriticNetwork
from stable_baselines.common.atari_wrappers import FrameStack

def run_episode(env, model):
    done = False
    episode_reward = 0.0
    obs = env.reset()

    while not done:
        state = obs._force()
        state = torch.Tensor(state.reshape(1, 4, 84, 84)).to(device)
        action_probs, value_ext, value_int = model(state)
        action_dist = Categorical(action_probs)
        action = int(action_dist.sample())

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
    env = FrameStack(env, 4)

    device = torch.device('cuda')

    input_size = env.observation_space.shape
    output_size = env.action_space.n

    model_path = 'rnd/trained_models/main.model'
    model = CnnActorCriticNetwork(input_size, output_size, False)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    if env.is_grading():
        episode_reward = run_evaluation(env, model)
    else:
        while True:
            episode_reward = run_episode(env, model)
            print("Episode reward: " + str(episode_reward))
            env.reset()

    env.close()


