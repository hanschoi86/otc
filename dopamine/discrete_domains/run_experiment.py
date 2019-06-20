# coding=utf-8
# Copyright 2018 The Dopamine Authors.
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
"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import unity_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import gin.tf


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_agent(sess, environment, agent_name=None, summary_writer=None,
                 debug_mode=False):
  """Creates an agent.

  Args:
    sess: A `tf.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  if not debug_mode:
    summary_writer = None
  if agent_name == 'dqn':
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'implicit_quantile':
    return implicit_quantile_agent.ImplicitQuantileAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
def create_runner(base_dir, schedule='continuous_train_and_eval'):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return Runner(base_dir, create_agent)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return TrainRunner(base_dir, create_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


@gin.configurable
class Runner(object):
  """Object that handles running Dopamine experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.

  A simple scenario to train a DQN agent is as follows:

  ```python
  import dopamine.discrete_domains.atari_lib
  base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
  runner = Runner(base_dir, create_agent, atari_lib.create_atari_environment)
  runner.run()
  ```
  """

  def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=unity_lib.create_otc_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=27000,
               key_reward=0,
               enable_front_rnd=False,
               intrinsic_batchsize=128,
               nonepisodic=False,
               pure_exploration=False):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a `tf.Session`.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    assert base_dir is not None
    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._create_directories()
    self._summary_writer = tf.summary.FileWriter(self._base_dir)

    self._environment = create_environment_fn()
    # Set up a session and initialize variables.
    self._sess = tf.Session('',
                            config=tf.ConfigProto(allow_soft_placement=True))
    self._agent = create_agent_fn(self._sess, self._environment,
                                  summary_writer=self._summary_writer)
    self._summary_writer.add_graph(graph=tf.get_default_graph())
    self._sess.run(tf.global_variables_initializer())

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)
    self.key_reward = key_reward
    self.current_key = 0

    # Enable Random Network Distillation (First version that adds intrinsic reward
    # before the experience replay storage. I will refer to this as front-RND
    self.enable_front_rnd = enable_front_rnd
    if self.enable_front_rnd:
      # Settings for front-RND
      self.intrinsic_target_network = self._build_intrinsic_target_network()
      self.intrinsic_predict_network = self._build_intrinsic_predict_network()
      self.intrinsic_batchsize = intrinsic_batchsize
      self.pure_exploration = pure_exploration
      self.intrinsic_std = None
      self.intrinsic_obs_running_mean = None
    self.nonepisodic = nonepisodic

  def _create_directories(self):
    """Create necessary sub-directories."""
    self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
    self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists.

    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoint in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the agent is able to successfully unbundle, this method will verify that
    the unbundled data contains the keys,'logs' and 'current_iteration'. It will
    then load the `Logger`'s data from the bundle, and will return the iteration
    number keyed by 'current_iteration' as one of the return values (along with
    the `Checkpointer` object).

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.

    Returns:
      start_iteration: int, the iteration number to start the experiment from.
      experiment_checkpointer: `Checkpointer` object for the experiment.
    """
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   checkpoint_file_prefix)
    self._start_iteration = 0
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        self._checkpoint_dir)
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      if self._agent.unbundle(
          self._checkpoint_dir, latest_checkpoint_version, experiment_data):
        if experiment_data is not None:
          assert 'logs' in experiment_data
          assert 'current_iteration' in experiment_data
          self._logger.data = experiment_data['logs']
          self._start_iteration = experiment_data['current_iteration'] + 1
        tf.logging.info('Reloaded checkpoint and will start from iteration %d',
                        self._start_iteration)

  def _initialize_episode(self):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    return self._agent.begin_episode(initial_observation)

  def _run_one_step(self, action):
    """Executes a single step in the environment.

    Args:
      action: int, the action to perform in the environment.

    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
    observation, reward, is_terminal, info = self._environment.step(action)
    # Reshape key reward from +0.1 to +0.8 (can be adjusted by argument)
    if self.current_key < info['total_keys']:
      reward += self.key_reward
    self.current_key = info['total_keys']
    return observation, reward, is_terminal

  def _end_episode(self, reward):
    """Finalizes an episode run.

    Args:
      reward: float, the last reward from the environment.
    """
    self._agent.end_episode(reward)

  def _run_one_episode(self, run_mode_str):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    action = self._initialize_episode()
    is_terminal = False

    # Initialize front-RND variables
    intrinsic_sum = 0
    intrinsic_count = 0
    intrinsic_series = np.array([])
    intrinsic_obs = None

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      step_number += 1

      # Perform reward clipping.
      reward = np.clip(reward, -1, 1)

      # Add front-RND
      if self.enable_front_rnd and (run_mode_str != 'eval'):
        reshaped_observation = np.expand_dims(observation, 0) / 255.
        # Observation Normalization with a clip of 5, for front-RND
        if intrinsic_obs is None:
          intrinsic_obs_series = reshaped_observation
        else:
          intrinsic_obs_series = np.append(intrinsic_obs_series, reshaped_observation, axis=0)

        if self.intrinsic_obs_running_mean is not None:
          self.intrinsic_obs_running_mean = intrinsic_obs_series.mean(axis=0)
          self.intrinsic_obs_running_std = intrinsic_obs_series.std(axis=0)
          normalized_obs = (reshaped_observation - self.intrinsic_obs_running_mean) / (self.intrinsic_obs_running_std + 10e-7)
          normalized_obs = np.clip(normalized_obs, -5, 5)
        else:
          normalized_obs = reshaped_observation

        # Batched training for front-RND
        if intrinsic_count == 0:
          batch_observation = normalized_obs
        else:
          batch_observation = np.append(batch_observation, normalized_obs, axis=0)
        intrinsic_count += 1

        # Calculate intrinsic reward
        intrinsic_target_value = self.intrinsic_target_network.predict(normalized_obs)
        intrinsic_predict_value = self.intrinsic_predict_network.predict(normalized_obs)
        intrinsic_reward = ((intrinsic_target_value - intrinsic_predict_value) ** 2).mean()

        # Intrinsic Reward Normalization by Running Std, for front-RND
        if len(intrinsic_series) < self.intrinsic_batchsize:
          intrinsic_series = np.append(intrinsic_series, intrinsic_reward)
        else:
          intrinsic_series = np.append(intrinsic_series, intrinsic_reward)[1:]
          self.intrinsic_std = intrinsic_series.std()
        if self.intrinsic_std is not None:
          intrinsic_reward /= self.intrinsic_std

        # Pure exploration = True will exclude extrinsic reward
        if self.pure_exploration:
          reward = intrinsic_reward
        else:
          reward += intrinsic_reward
        intrinsic_sum += intrinsic_reward

        # Update predictor network for front
        if intrinsic_count == self.intrinsic_batchsize:
          intrinsic_count = 0
          self.intrinsic_predict_network.fit(batch_observation,
                                             np.tile(intrinsic_target_value, (self.intrinsic_batchsize, 1)),
                                             batch_size=self.intrinsic_batchsize, verbose=0)

      # Episode termination for front-RND (add nonepisodic functionality)

      if ((self._environment.game_over and not self.nonepisodic) or
          (self._environment.game_over and (run_mode_str == 'eval')) or
          (step_number == self._max_steps_per_episode)):
        # Stop the run loop once we reach the true end of episode.
        break
      elif ((is_terminal and not self.nonepisodic) or
            (is_terminal and (run_mode_str == 'eval'))):
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._agent.end_episode(reward)
        action = self._agent.begin_episode(observation)
      else:
        action = self._agent.step(reward, observation)

    # front-RND logging
    self._end_episode(reward)

    if self.enable_front_rnd:
      intrinsic_average = intrinsic_sum / step_number
    else:
      intrinsic_average = np.nan
    return step_number, total_reward, intrinsic_average

  def _build_intrinsic_target_network(self):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(84, 84, 1)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation=None))
    model.trainable = False
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.build()
    return model

  def _build_intrinsic_predict_network(self):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(84, 84, 1)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation=None))
    model.trainable = True
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.build()
    return model

  def _run_one_phase(self, min_steps, statistics, run_mode_str):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    while step_count < min_steps:
      episode_length, episode_return, intrinsic_average = self._run_one_episode(run_mode_str)
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      # We use sys.stdout.write instead of tf.logging so as to flush frequently
      # without generating a line break.
      if self.enable_front_rnd:
        sys.stdout.write('Steps executed: {} '.format(step_count) +
                         'Episode length: {} '.format(episode_length) +
                         'Intrinsic Reward: {} '.format(intrinsic_average) +
                         'Return: {}\r'.format(episode_return))
      else:
        sys.stdout.write('Steps executed: {} '.format(step_count) +
                         'Episode length: {} '.format(episode_length) +
                         'Return: {}\r'.format(episode_return))
      sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def _run_train_phase(self, statistics):
    """Run training phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: The average reward generated in this phase.
    """
    # Perform the training phase, during which the agent learns.
    self._agent.eval_mode = False
    start_time = time.time()
    number_steps, sum_returns, num_episodes = self._run_one_phase(
        self._training_steps, statistics, 'train')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    statistics.append({'train_average_return': average_return})
    time_delta = time.time() - start_time
    tf.logging.info('Average undiscounted return per training episode: %.2f',
                    average_return)
    tf.logging.info('Average training steps per second: %.2f',
                    number_steps / time_delta)
    return num_episodes, average_return

  def _run_eval_phase(self, statistics):
    """Run evaluation phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    _, sum_returns, num_episodes = self._run_one_phase(
        self._evaluation_steps, statistics, 'eval')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    tf.logging.info('Average undiscounted return per evaluation episode: %.2f',
                    average_return)
    statistics.append({'eval_average_return': average_return})
    return num_episodes, average_return

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. The interleaving of train/eval phases implemented here
    are to match the implementation of (Mnih et al., 2015).

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    num_episodes_train, average_reward_train = self._run_train_phase(
        statistics)
    num_episodes_eval, average_reward_eval = self._run_eval_phase(
        statistics)

    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train, num_episodes_eval,
                                     average_reward_eval)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_train,
                                  average_reward_train,
                                  num_episodes_eval,
                                  average_reward_eval):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_train: int, number of training episodes run.
      average_reward_train: float, The average training reward.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
    """
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Train/NumEpisodes',
                         simple_value=num_episodes_train),
        tf.Summary.Value(tag='Train/AverageReturns',
                         simple_value=average_reward_train),
        tf.Summary.Value(tag='Eval/NumEpisodes',
                         simple_value=num_episodes_eval),
        tf.Summary.Value(tag='Eval/AverageReturns',
                         simple_value=average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)

  def _log_experiment(self, iteration, statistics):
    """Records the results of the current iteration.

    Args:
      iteration: int, iteration number.
      statistics: `IterationStatistics` object containing statistics to log.
    """
    self._logger['iteration_{:d}'.format(iteration)] = statistics
    if iteration % self._log_every_n == 0:
      self._logger.log_to_file(self._logging_file_prefix, iteration)

  def _checkpoint_experiment(self, iteration):
    """Checkpoint experiment data.

    Args:
      iteration: int, iteration number for checkpointing.
    """
    experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                        iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      experiment_data['logs'] = self._logger.data
      self._checkpointer.save_checkpoint(iteration, experiment_data)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    tf.logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                         self._num_iterations, self._start_iteration)
      return

    for iteration in range(self._start_iteration, self._num_iterations):
      statistics = self._run_one_iteration(iteration)
      self._log_experiment(iteration, statistics)
      self._checkpoint_experiment(iteration)


@gin.configurable
class TrainRunner(Runner):
  """Object that handles running experiments.

  The `TrainRunner` differs from the base `Runner` class in that it does not
  the evaluation phase. Checkpointing and logging for the train phase are
  preserved as before.
  """

  def __init__(self, base_dir, create_agent_fn,
               create_environment_fn=unity_lib.create_otc_environment):
    """Initialize the TrainRunner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
    """
    tf.logging.info('Creating TrainRunner ...')
    super(TrainRunner, self).__init__(base_dir, create_agent_fn,
                                      create_environment_fn)
    self._agent.eval_mode = False

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. This method differs from the `_run_one_iteration` method
    in the base `Runner` class in that it only runs the train phase.

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    num_episodes_train, average_reward_train = self._run_train_phase(
        statistics)

    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward):
    """Save statistics as tensorboard summaries."""
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Train/NumEpisodes', simple_value=num_episodes),
        tf.Summary.Value(
            tag='Train/AverageReturns', simple_value=average_reward),
    ])
    self._summary_writer.add_summary(summary, iteration)
