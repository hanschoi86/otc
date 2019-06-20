# Load libraries
import numpy as np
import gin.tf
from obstacle_tower_env import ActionFlattener
from gym.spaces.box import Box

# Preprocess environment
@gin.configurable
class Preprocessing(object):
  """A class implementing image preprocessing for OTC agents.

  Specifically, this converts observations to greyscale. It doesn't
  do anything else to the environment.
  """

  def __init__(self, environment, reshape_action=False, fetch_key=False):
    """Constructor for an Obstacle Tower preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.

    """
    self.environment = environment
    self.fetch_key = fetch_key

    self.game_over = False
    self.lives = 0  # Will need to be set by reset().

    if reshape_action:
      self.environment._flattener = ActionFlattener([2, 3, 2, 1])
      self.environment._action_space = self.environment._flattener.action_space

  @property
  def observation_space(self):
    if self.fetch_key:
      return Box(0, 255, shape=(7064,))
    else:
      return self.environment.observation_space

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

  def _flatten_(self, obs, key):
    return

  def reset(self):
    observation = self.environment.reset()
    observation = observation / 255.
    if self.fetch_key:
      observation = np.append(observation, [1, 0, 0, 0, 0, 0, 1, 0])
    return observation

  def render(self, mode):
    return self.environment.render(mode)

  def step(self, action):
    observation, reward, game_over, info = self.environment.step(action)
    self.game_over = game_over
    observation = observation / 255.
    if self.fetch_key:
      key_obs = info['brain_info'].vector_observations
      key_obs[-1, -2] = key_obs[-1, -2] / 3000
      observation = np.append(observation, key_obs)
    return observation, reward, game_over, info
