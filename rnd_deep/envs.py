import gym

import numpy as np

from collections import deque
from copy import copy

from torch.multiprocessing import Pipe, Process
from otcenv.obstacle_tower_env import ObstacleTowerEnv, ActionFlattener

from model import *
from PIL import Image

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.is_render = is_render

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if self.is_render:
                self.env.render()
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        

class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def _unwrap(self, env):
        if hasattr(env, "unwrapped"):
            return env.unwrapped
        elif hasattr(env, "env"):
            return unwrap(env.env)
        elif hasattr(env, "leg_env"):
            return unwrap(env.leg_env)
        else:
            return env

    def get_current_room(self):
        ram = self._unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())

        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_rooms=copy(self.visited_rooms))

        if done:
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()


class AtariEnvironment(Process):
    def __init__(
            self,
            env_name,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84,
            sticky_action=True,
            p=0.25,
            max_episode_steps=18000):
        super(AtariEnvironment, self).__init__()
        self.seed = np.random.randint(0, 60000)
        self.daemon = True
        self.env = ObstacleTowerEnv('../ObstacleTower/obstacletower', worker_id=self.seed,
                               retro=True, config={'total-floors': 10}, greyscale=True, timeout_wait=300)
        self.env._flattener = ActionFlattener([2, 3, 2, 1])
        self.env._action_space = self.env._flattener.action_space
        self.env_name = env_name
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p
        self.max_episode_steps = max_episode_steps

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super(AtariEnvironment, self).run()
        while True:
            action = self.child_conn.recv()

            if 'Breakout' in self.env_name:
                action += 1

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action

            s, reward, done, info = self.env.step(action)

            if self.max_episode_steps < self.steps:
                done = True

            log_reward = reward
            force_done = done

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(s)

            self.rall += reward
            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}".format(
                    self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist)))

                self.history = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], reward, force_done, done, log_reward])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s = self.env.reset()
        self.get_init_state(
            self.pre_proc(s))
        return self.history[:, :, :]

    def pre_proc(self, X):
        #X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        #x = cv2.resize(X, (self.h, self.w))
        #return x
        frame = Image.fromarray(X.reshape(84, 84)).convert('L')
        frame = np.array(frame.resize((self.h, self.w)))
        return frame.astype(np.float32)

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)
