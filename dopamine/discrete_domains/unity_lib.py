# coding=utf-8
# Copyright 2018 The Dopamine Authors.
# Modifications copyright 2019 Unity Technologies.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Obstacle Tower-specific utilities including Atari-specific network architectures.

This includes a class implementing minimal preprocessing, which
is in charge of:
  . Converting observations to greyscale.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from obstacle_tower_env import ObstacleTowerEnv, ActionFlattener
from gym.spaces.box import Box

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from dopamine.discrete_domains import inception, inception_utils, inception_resnet
from gym import spaces

import gin.tf

slim = tf.contrib.slim


NATURE_DQN_OBSERVATION_SHAPE = (84, 84)  # Size of downscaled Atari 2600 frame.
NATURE_DQN_DTYPE = tf.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 4  # Number of frames in the state stack.

class CustomActionFlattener():
    def __init__(self):
        self._action_shape = [2, 3, 2, 1]
        self.action_lookup = {0: [0, 0, 0, 0], 1: [0, 0, 1, 0], 2: [0, 1, 0, 0], 3: [0, 2, 0, 0],
                              4: [1, 0, 0, 0], 5: [1, 0, 1, 0], 6: [1, 1, 0, 0], 7: [1, 2, 0, 0],
                              8: [1, 1, 1, 0], 9: [1, 2, 1, 0]}
        self.action_space = spaces.Discrete(len(self.action_lookup))

    def lookup_action(self, action):
        return self.action_lookup[action]

@gin.configurable
def create_otc_environment(environment_path=None, custom_action=True, start_floor=0, fetch_key=False,
                           enable_obsnorm=False, obsnorm_mode='pixel', obsnorm_period=10000):
  """Wraps an Obstacle Tower Gym environment with some basic preprocessing.

  Returns:
    An Obstacle Tower environment with some standard preprocessing.
  """
  assert environment_path is not None
  env_config = {"total-floors": 15, "dense-reward": 1, "starting-floor": start_floor, 'visual-theme': 0.0}
  env = ObstacleTowerEnv(environment_path, worker_id=np.random.randint(100, 300), retro=True, config=env_config, greyscale=True)

  env = OTCPreprocessing(env, fetch_key=fetch_key, custom_action=custom_action, enable_obsnorm=enable_obsnorm, obsnorm_mode=obsnorm_mode, obsnorm_period=obsnorm_period)
  return env



@gin.configurable
class OTCPreprocessing(object):
  """A class implementing image preprocessing for OTC agents.

  Specifically, this converts observations to greyscale. It doesn't
  do anything else to the environment.
  """

  def __init__(self, environment, fetch_key=False, enable_obsnorm=False, obsnorm_mode='pixel',
               obsnorm_period=10000, custom_action=True):
    """Constructor for an Obstacle Tower preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.

    """
    self.environment = environment

    self.game_over = False
    self.lives = 0  # Will need to be set by reset().

    self.fetch_key = fetch_key
    # Enable observation normalization
    self.enable_obsnorm = enable_obsnorm

    # obsnorm_mode='pixel' is normalization by pixel, 'pic' is by entire image
    self.obsnorm_mode = obsnorm_mode
    self.obsnorm_period = obsnorm_period
    self.custom_action = custom_action

    if self.enable_obsnorm:
      self.running_obs = np.random.random(self.environment.observation_space.shape)
      self.obs_counter = 0

    if self.custom_action:
      self.environment._flattener = CustomActionFlattener()
    else:
      self.environment._flattener = ActionFlattener([2, 3, 2, 1])
    self.environment._action_space = self.environment._flattener.action_space

  def normalize_observation(self, observation, obsnorm_mode, obsnorm_period):
      epsilon = 10e-5
      self.obs_counter += 1
      observation = observation / 255.
      self.running_obs = np.append(self.running_obs, observation, axis=2)[:, :, -obsnorm_period:]
      
      if obsnorm_mode == 'pixel':
        pixel_mean = self.running_obs.mean(axis=2, keepdims=True)
        pixel_std = self.running_obs.std(axis=2, keepdims=True)
        observation = (observation - pixel_mean) / (pixel_std + epsilon)
      elif obsnorm_mode == 'pic':
        pixel_mean = self.running_obs.mean()
        pixel_std = self.running_obs.std()
        observation = (observation - pixel_mean) / (pixel_std + epsilon)
      return observation
      
  @property
  def observation_space(self):
    return self.environment.observation_space
    #return Box(0, 255, shape=(7064,))

  @property
  def spec(self):
    return self.environment.spec
    
  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  @property
  def seed(self):
      return self.environment.seed

  def reset(self):
    """Resets the environment. Converts the observation to greyscale, 
    if it is not. 

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
    #if(len(observation.shape)> 2):
    #  observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = self.environment.reset()
    observation = observation / 255.
    if self.fetch_key:
      observation = np.append(observation, [1, 0, 0, 0, 0, 0, 1, 0])
    if self.enable_obsnorm:
      observation = self.normalize_observation(observation, self.obsnorm_mode, self.obsnorm_period)
    return observation

  def render(self, mode):
    """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
    return self.environment.render(mode)

  def step(self, action):
    """Applies the given action in the environment. Converts the observation to 
    greyscale, if it is not. 

    Remarks:

      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """

    #if(len(observation.shape)> 2):
    #  observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

    observation, reward, game_over, info = self.environment.step(action)
    self.game_over = game_over
    observation = observation / 255.
    if self.fetch_key:
      key_obs = info['brain_info'].vector_observations
      key_obs[-1, -2] = key_obs[-1, -2] / 3000
      observation = np.append(observation, key_obs)
    if self.enable_obsnorm:
      observation = self.normalize_observation(observation, self.obsnorm_mode, self.obsnorm_period)
    return observation, reward, game_over, info

def nature_dqn_network(num_actions, network_type, state):
  """The convolutional network used to compute the agent's Q-values.

  Args:
    num_actions: int, number of actions.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  net = tf.cast(state, tf.float32)
  net = slim.conv2d(net, 32, [8, 8], stride=4)
  net = slim.conv2d(net, 64, [4, 4], stride=2)
  net = slim.conv2d(net, 64, [3, 3], stride=1)
  net = slim.flatten(net)
  net = slim.fully_connected(net, 512)
  q_values = slim.fully_connected(net, num_actions, activation_fn=None)
  return network_type(q_values)


@gin.configurable
def rainbow_network(num_actions, num_atoms, support, network_type, state):
  """The convolutional network used to compute agent's Q-value distributions.

  Args:
    num_actions: int, number of actions.
    num_atoms: int, the number of buckets of the value function distribution.
    support: tf.linspace, the support of the Q-value distribution.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  net = tf.cast(state, tf.float32)
  #scaled = tf.slice(net, [0, 0, 0], [-1, 84 * 84, -1])
  #scaled = tf.reshape(scaled, shape=(-1, 84, 84, 4))
  #keys = tf.slice(net, [0, 84 * 84, 0], [-1, -1, -1])

  net = slim.conv2d(
      net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
  net = slim.conv2d(
      net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
  net = slim.conv2d(
      net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
  net = slim.flatten(net)
  #keys = slim.flatten(keys)
  #net = tf.concat([net, keys], axis=1)
  net = slim.fully_connected(
      net, 512, weights_initializer=weights_initializer)
  net = slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer)

  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = tf.contrib.layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities)

@gin.configurable
def rainbow_network_key(num_actions, num_atoms, support, network_type, state):
  """The convolutional network used to compute agent's Q-value distributions.

  Args:
    num_actions: int, number of actions.
    num_atoms: int, the number of buckets of the value function distribution.
    support: tf.linspace, the support of the Q-value distribution.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  net = tf.cast(state, tf.float32)
  scaled = tf.slice(net, [0, 0, 0], [-1, 84 * 84, -1])
  scaled = tf.reshape(scaled, shape=(-1, 84, 84, 4))
  keys = tf.slice(net, [0, 84 * 84, 0], [-1, -1, -1])

  net = slim.conv2d(
      scaled, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
  net = slim.conv2d(
      net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
  net = slim.conv2d(
      net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
  net = slim.flatten(net)
  keys = slim.flatten(keys)
  net = tf.concat([net, keys], axis=1)
  net = slim.fully_connected(
      net, 512, weights_initializer=weights_initializer)
  net = slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer)

  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = tf.contrib.layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities)

@gin.configurable
def rainbow_network_deep(num_actions, num_atoms, support, network_type, state):
  # Deeper version of Nature's deep NN architecture
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  net = tf.cast(state, tf.float32)
  net = slim.conv2d(
      net, 16, [8, 8], stride=2, weights_initializer=weights_initializer)
  net = slim.conv2d(
      net, 32, [6, 6], stride=2, weights_initializer=weights_initializer)
  net = slim.conv2d(
      net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
  net = slim.conv2d(
      net, 128, [3, 3], stride=1, weights_initializer=weights_initializer)
  net = slim.flatten(net)
  net = slim.fully_connected(
      net, 512, weights_initializer=weights_initializer)
  net = slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer)

  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = tf.contrib.layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities)


@gin.configurable
def rainbow_network_lenet(num_actions, num_atoms, support, network_type, state):
  # Lenet-like network
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  net = tf.cast(state, tf.float32)
  net = slim.conv2d(
      net, 6, [5, 5], stride=1, weights_initializer=weights_initializer, padding='VALID') #80x80x6
  net = slim.avg_pool2d(net, [2, 2], stride=2) #40x40x6
  net = slim.conv2d(
      net, 16, [5, 5], stride=1, weights_initializer=weights_initializer, padding='VALID') #76x76x16
  net = slim.avg_pool2d(net, [2, 2], stride=2) #38x38x16
  net = slim.conv2d(
      net, 32, [5, 5], stride=1, weights_initializer=weights_initializer, padding='VALID')  # 34x34x32
  net = slim.avg_pool2d(net, [2, 2], stride=2)  # 17x17x32
  net = slim.flatten(net)
  net = slim.fully_connected(
      net, 256, weights_initializer=weights_initializer)
  net = slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer)

  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = tf.contrib.layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities)


@gin.configurable
def rainbow_network_alexnet(num_actions, num_atoms, support, network_type, state):
  # Alexnet-like network
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  net = tf.cast(state, tf.float32)
  net = slim.conv2d(
      net, 48, [8, 8], stride=2, weights_initializer=weights_initializer, padding='VALID') #39x39x96
  net = slim.max_pool2d(net, [3, 3], stride=2) #19x19x96
  net = slim.conv2d(
      net, 128, [5, 5], stride=1, weights_initializer=weights_initializer)  #19x19x128
  net = slim.max_pool2d(net, [3, 3], stride=2)  #9x9x128
  net = slim.conv2d(
      net, 192, [3, 3], stride=1, weights_initializer=weights_initializer)  # 9x9x192
  net = slim.conv2d(
      net, 192, [3, 3], stride=1, weights_initializer=weights_initializer)  # 9x9x192
  net = slim.conv2d(
      net, 128, [3, 3], stride=1, weights_initializer=weights_initializer)  # 9x9x128
  net = slim.max_pool2d(net, [3, 3], stride=2)  # 4x4x128
  net = slim.flatten(net)
  net = slim.fully_connected(
      net, 1024, weights_initializer=weights_initializer)
  net = slim.fully_connected(
      net, 1024, weights_initializer=weights_initializer)
  net = slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer)

  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = tf.contrib.layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities)


@gin.configurable
def rainbow_network_vgg(num_actions, num_atoms, support, network_type, state):
  # VGG-like network
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)
  net = tf.cast(state, tf.float32)
  net = slim.conv2d(
      net, 64, [3, 3], stride=1, weights_initializer=weights_initializer) # 84x84x64
  net = slim.max_pool2d(net, [2, 2], stride=2)  # 42x42x64
  net = slim.conv2d(
      net, 128, [3, 3], stride=1, weights_initializer=weights_initializer)  # 42x42x128
  net = slim.max_pool2d(net, [2, 2], stride=2)  # 21x21x128
  net = slim.conv2d(
      net, 256, [3, 3], stride=1, weights_initializer=weights_initializer)  # 21x21x256
  net = slim.conv2d(
      net, 256, [3, 3], stride=1, weights_initializer=weights_initializer)  # 21x21x256
  net = slim.max_pool2d(net, [2, 2], stride=2)  # 10x10x256
  net = slim.conv2d(
      net, 512, [3, 3], stride=1, weights_initializer=weights_initializer)  # 10x10x512
  net = slim.conv2d(
      net, 512, [3, 3], stride=1, weights_initializer=weights_initializer)  # 10x10x512
  net = slim.max_pool2d(net, [2, 2], stride=2)  # 5x5x512
  net = slim.flatten(net)
  net = slim.fully_connected(
      net, 2048, weights_initializer=weights_initializer)
  net = slim.fully_connected(
      net, 2048, weights_initializer=weights_initializer)
  net = slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer)

  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = tf.contrib.layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities)


@gin.configurable
def rainbow_network_resnet(num_actions, num_atoms, support, network_type, state, is_training=False):
  # Embedded Resnet50 block from TF Slim
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)
  net = tf.cast(state, tf.float32)
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_50(net,
                                               num_classes=None,
                                                is_training=is_training,
                                                global_pool=False,
                                                output_stride=None)
  net = slim.flatten(net)
  net = slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer)

  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = tf.contrib.layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities)


@gin.configurable
def rainbow_network_inception(num_actions, num_atoms, support, network_type, state, is_training=False):
  # Embedded Inception block from TF Slim
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)
  net = tf.cast(state, tf.float32)
  with slim.arg_scope(inception_utils.inception_arg_scope()):
      net, end_points = inception.inception_v3(net,
                                               num_classes=None,
                                               is_training=is_training)
  net = slim.flatten(net)
  net = slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer)

  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = tf.contrib.layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities)


@gin.configurable
def rainbow_network_inception_resnet(num_actions, num_atoms, support, network_type, state, is_training=False):
  # Embedded Inception-Resnet block from TF Slim
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)
  net = tf.cast(state, tf.float32)
  with slim.arg_scope(inception_resnet.inception_resnet_v2_arg_scope()):
      net, end_points = inception_resnet.inception_resnet_v2(net,
                                                             num_classes=None,
                                                             is_training=is_training)
  net = slim.flatten(net)
  net = slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer)

  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = tf.contrib.layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities)


def implicit_quantile_network(num_actions, quantile_embedding_dim,
                              network_type, state, num_quantiles):
  """The Implicit Quantile ConvNet.

  Args:
    num_actions: int, number of actions.
    quantile_embedding_dim: int, embedding dimension for the quantile input.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.
    num_quantiles: int, number of quantile inputs.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  state_net = tf.cast(state, tf.float32)
  state_net = slim.conv2d(
      state_net, 32, [8, 8], stride=4,
      weights_initializer=weights_initializer)
  state_net = slim.conv2d(
      state_net, 64, [4, 4], stride=2,
      weights_initializer=weights_initializer)
  state_net = slim.conv2d(
      state_net, 64, [3, 3], stride=1,
      weights_initializer=weights_initializer)
  state_net = slim.flatten(state_net)
  state_net_size = state_net.get_shape().as_list()[-1]
  state_net_tiled = tf.tile(state_net, [num_quantiles, 1])

  batch_size = state_net.get_shape().as_list()[0]
  quantiles_shape = [num_quantiles * batch_size, 1]
  quantiles = tf.random_uniform(
      quantiles_shape, minval=0, maxval=1, dtype=tf.float32)

  quantile_net = tf.tile(quantiles, [1, quantile_embedding_dim])
  pi = tf.constant(math.pi)
  quantile_net = tf.cast(tf.range(
      1, quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
  quantile_net = tf.cos(quantile_net)
  quantile_net = slim.fully_connected(quantile_net, state_net_size,
                                      weights_initializer=weights_initializer)
  # Hadamard product.
  net = tf.multiply(state_net_tiled, quantile_net)

  net = slim.fully_connected(
      net, 512, weights_initializer=weights_initializer)
  quantile_values = slim.fully_connected(
      net,
      num_actions,
      activation_fn=None,
      weights_initializer=weights_initializer)

  return network_type(quantile_values=quantile_values, quantiles=quantiles)

