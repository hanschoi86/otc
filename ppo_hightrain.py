import numpy as np
import os
import argparse
from obstacle_tower_env import ObstacleTowerEnv, ActionFlattener

from deep_baselines.common.policies import MlpPolicy, CnnPolicy
from deep_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from deep_baselines import PPO2
from otc_prep import Preprocessing

seed = np.random.randint(0, 100)

def make_env(log_dir, cpu):
    sub_dir = "{}cpu_{}/".format(log_dir, cpu)
    os.makedirs(sub_dir, exist_ok=True)
    def _init():
        env = ObstacleTowerEnv('./ObstacleTower/obstacletower', worker_id=seed+cpu,
                               retro=False, config={'total-floors': 10}, greyscale=False, timeout_wait=600)
        env._flattener = ActionFlattener([2, 3, 2, 1])
        env._action_space = env._flattener.action_space
        env = Preprocessing(env)
        env = Monitor(env, sub_dir)
        return env
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_cpu', help='number of cpu cores', type=int, default=16)
    parser.add_argument('--gamma', help='PPO gamma', type=float, default=0.999)
    parser.add_argument('--num_timesteps', type=int, default=int(2e7))
    parser.add_argument('--learning_rate', type=float, default=.000135)
    args = parser.parse_args()

    num_cpu = args.num_cpu
    log_dir = "models/ppohigh/"
    os.makedirs(log_dir, exist_ok=True)
    multienv = SubprocVecEnv([make_env(log_dir, cpu) for cpu in range(num_cpu)])

    multimodel = PPO2(CnnPolicy, multienv, verbose=1, gamma=args.gamma, learning_rate=args.learning_rate, n_steps=128)
    multimodel = multimodel.load('models/ppohigh/highmodel', multienv)
    multimodel.learn(total_timesteps=args.num_timesteps)
    multimodel.save('models/ppohigh/highmodel')
