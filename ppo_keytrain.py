import numpy as np
import os
import argparse
from obstacle_tower_env import ObstacleTowerEnv, ActionFlattener
from otc_prep import Preprocessing
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import PPO2

seed = np.random.randint(0, 100)

def make_env(log_dir, cpu):
    sub_dir = "{}cpu_{}/".format(log_dir, cpu)
    os.makedirs(sub_dir, exist_ok=True)
    def _init():
        env = ObstacleTowerEnv('./ObstacleTower/obstacletower', worker_id=seed+cpu,
                               retro=True, config={'starting-floor': 5, 'total-floors': 11}, greyscale=True, timeout_wait=600)
        env._flattener = ActionFlattener([2, 3, 2, 1])
        env._action_space = env._flattener.action_space
        env = Preprocessing(env)
        env = Monitor(env, sub_dir)
        return env
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_cpu', help='number of cpu cores', type=int, default=12)
    parser.add_argument('--gamma', help='PPO gamma', type=float, default=0.999)
    parser.add_argument('--num_timesteps', type=int, default=int(3e7))
    parser.add_argument('--learning_rate', type=float, default=.0000625)
    args = parser.parse_args()

    num_cpu = args.num_cpu
    log_dir = "models/ppo/key/"
    os.makedirs(log_dir, exist_ok=True)
    # Create vectorized multi-environment to run on 12 cores
    multienv = SubprocVecEnv([make_env(log_dir, cpu) for cpu in range(num_cpu)])

    # Create PPO model for GPU
    multimodel = PPO2(CnnPolicy, multienv, verbose=1, gamma=args.gamma, learning_rate=args.learning_rate)
    multimodel = multimodel.load('models/ppo/multimodel3', multienv)
    multimodel.learn(total_timesteps=args.num_timesteps)
    multimodel.save('models/ppo/key/keymodel')
