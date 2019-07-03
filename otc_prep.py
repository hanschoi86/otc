# Load libraries
import numpy as np
import gin.tf
from PIL import Image
from gym.spaces import Box

# Preprocess environment
@gin.configurable
class Preprocessing(object):
  """A class implementing image preprocessing for OTC agents.

  Specifically, this converts observations to greyscale. It doesn't
  do anything else to the environment.
  """

  def __init__(self, environment, depth=3):
    """Constructor for an Obstacle Tower preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.

    """
    self.environment = environment
    self.depth = depth
    self.game_over = False
    self.lives = 0  # Will need to be set by reset().

  @property
  def observation_space(self):
    return Box(
            0, 255,
            dtype=np.float32,
            shape=(84, 84, self.depth))

  @property
  def _observation_space(self):
    return Box(
            0, 255,
            dtype=np.float32,
            shape=(84, 84, self.depth))
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
    obs_image = Image.fromarray(np.array(observation[0] * 255, dtype=np.uint8))
    obs_image = obs_image.resize((84, 84), Image.NEAREST)
    observation = np.array(obs_image)
    return observation

  def render(self, mode):
    return self.environment.render(mode)

  def step(self, action):
    observation, reward, game_over, info = self.environment.step(action)
    self.game_over = game_over

    obs_image = Image.fromarray(np.array(observation[0] * 255, dtype=np.uint8))
    obs_image = obs_image.resize((84, 84), Image.NEAREST)
    observation = np.array(obs_image)
    return observation, reward, game_over, info
