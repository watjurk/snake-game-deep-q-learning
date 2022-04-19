import os

import cv2
from ml_tools import q_learning


class Preprocessor(q_learning.Preprocessor):
  def __init__(self):
    self.current_folder = os.path.dirname(__file__)
    self.last_observations = None

  def reset(self):
    self.last_observations = None

  def preprocess(self, observation):
    assert(observation.shape == (150, 150, 3))

    if self.last_observations is not None:
      alpha = 0.6
      beta = (1.0 - alpha)
      observation = cv2.addWeighted(observation, alpha, self.last_observations, beta, 0)
    self.last_observations = observation

    observation = cv2.resize(observation, dsize=(50, 50), interpolation=cv2.INTER_AREA)
    self.ui.video.update_stream("preprocessed", observation)

    observation = observation / 255.0
    return observation
