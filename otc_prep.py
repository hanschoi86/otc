# Load libraries
import numpy as np
import gin.tf

# Preprocess environment
@gin.configurable
class Preprocessing(object):
  """A class implementing image preprocessing for OTC agents.

  Specifically, this converts observations to greyscale. It doesn't
  do anything else to the environment.
  """

  def __init__(self, environment):
    """Constructor for an Obstacle Tower preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.

    """
    self.environment = environment

    self.game_over = False
    self.lives = 0  # Will need to be set by reset().

  @property
  def observation_space(self):
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
    return observation

  def render(self, mode):
    return self.environment.render(mode)

  def step(self, action):
    observation, reward, game_over, info = self.environment.step(action)
    self.game_over = game_over
    if info['total_keys'] > 0:
      observation = 255 - observation
    return observation, reward, game_over, info
