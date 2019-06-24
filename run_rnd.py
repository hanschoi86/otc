import numpy as np
import argparse
from obstacle_tower_env import ObstacleTowerEnv, ActionFlattener

import torch
from torch.multiprocessing import Pipe
from torch.distributions.categorical import Categorical

from rnd.model import CnnActorCriticNetwork
from rnd.eval_atari import OTCEnvironment

def get_action(model, device, state):
    state = torch.Tensor(state).to(device)
    action_probs, value_ext, value_int = model(state)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()
    return action.data.cpu().numpy().squeeze()

def run_episode(model, device, parent_conn):
    done = False
    episode_reward = 0.0
    states = torch.zeros(num_worker, 4, 84, 84)

    while not done:
        actions = get_action(model, device, torch.div(states, 255.))
        parent_conn.send(actions)
        next_states = []
        next_state, reward, semi_done, done, log_reward = parent_conn.recv()
        next_states.append(next_state)
        states = torch.from_numpy(np.stack(next_states))
        states = states.type(torch.FloatTensor)

        episode_reward += reward
    return episode_reward


def run_evaluation(env, model, device, parent_conn):
    while not env.done_grading():
        run_episode(model, device, parent_conn)
        env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_filename', default='./ObstacleTower/obstacletower', nargs='?')
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    args = parser.parse_args()

    device = torch.device('cuda')

    input_size = (84, 84, 4)
    output_size = 12

    model_path = 'rnd/trained_models/main.model'
    num_worker = 1
    sticky_action = False

    model = CnnActorCriticNetwork(input_size, output_size, sticky_action)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    parent_conn, child_conn = Pipe()
    work = OTCEnvironment(
        'otc',
        False,
        0,
        child_conn,
        sticky_action=sticky_action,
        p=.25,
        max_episode_steps=18000)
    work.start()

    if work.env.is_grading():
        episode_reward = run_evaluation(work.env, model, device, parent_conn)
    else:
        while True:
            episode_reward = run_episode(model, device, parent_conn)
            print("Episode reward: " + str(episode_reward))
            env.reset()

    env.close()


