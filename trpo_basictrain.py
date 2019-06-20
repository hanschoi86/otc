import numpy as np
import os
import argparse
from obstacle_tower_env import ObstacleTowerEnv, ActionFlattener
from otc_prep import Preprocessing
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import TRPO

seed = np.random.randint(0, 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gamma', help='PPO gamma', type=float, default=0.999)
    parser.add_argument('--num_timesteps', type=int, default=int(3e7))
    args = parser.parse_args()

    log_dir = "models/trpo/"
    os.makedirs(log_dir, exist_ok=True)

    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', worker_id=seed,
                           retro=True, config={'total-floors': 8}, greyscale=True, timeout_wait=600)
    env._flattener = ActionFlattener([2, 3, 2, 1])
    env._action_space = env._flattener.action_space
    env = Preprocessing(env)
    env = Monitor(env, log_dir)


    multimodel = TRPO(CnnPolicy, env, verbose=1, gamma=args.gamma)
    multimodel.learn(total_timesteps=args.num_timesteps)
    multimodel.save('models/trpo/trpomodel')
