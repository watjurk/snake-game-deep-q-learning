"""
Basic brain
differences with previous:
- add frame stating
"""

from os import path

import cv2
from ml_tools import q_learning
from tensorflow import keras


class Brain(q_learning.Brain):
  def __init__(self): 
    self.model_path = path.join(path.dirname(__file__), 'model')
    self.extend_old_model = False

    self.last_observations = None
    self.input_shape=(38, 38, 3)
    self.batched_input_shape = (None, *self.input_shape)

  def save_model(self):
    self.model.save(self.model_path)

  def load_model(self):
    self.model = keras.models.load_model(self.model_path)

  def build_model(self, learning_rate):
    if self.extend_old_model:
      self.load_model()
      old_model = self.model
      old_layers = old_model.layers[:-1]
      
      new_model = keras.Sequential()
      for layer in old_layers:
        layer.trainable = False
        new_model.add(layer)
      
      new_model.add(keras.layers.Dense(128, activation='relu', name='new_dense_1'))
      new_model.add(keras.layers.Dense(64, activation='relu', name='new_dense_2'))
      new_model.add(keras.layers.Dense(4, activation='relu', name='new_dense_3'))

      new_model.build(self.batched_input_shape)
      self.model = new_model
    else:
      self.model = keras.Sequential()

      self.model.add(keras.layers.InputLayer(input_shape=self.input_shape))

      self.model.add(keras.layers.Convolution2D(64, kernel_size=(8, 8), activation='relu'))
      self.model.add(keras.layers.MaxPooling2D(pool_size=(4, 4)))

      self.model.add(keras.layers.Convolution2D(32, kernel_size=(4, 4), activation='relu'))
      self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

      self.model.add(keras.layers.Flatten())

      self.model.add(keras.layers.Dense(256, activation='relu'))

      self.model.add(keras.layers.Dense(4))

    self.model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

  def prepare_observation(self, observation):
    assert(observation.shape == (150, 150, 3))

    if self.last_observations is not None:
      alpha = 0.6
      beta = (1.0 - alpha)
      observation = cv2.addWeighted(observation, alpha, self.last_observations, beta, 0)
    self.last_observations = observation

    observation = cv2.resize(observation, dsize=(38, 38), interpolation=cv2.INTER_AREA)
    # self.ui.video.update_stream('brain', observation)
    
    observation = observation / 255.0
    return observation

  def reset_prepare(self):
    self.last_observations = None
