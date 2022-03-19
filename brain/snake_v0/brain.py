"""
Basic brain, without "frame stacking"
"""

from os import path

import cv2
from ml_tools import q_learning
from tensorflow import keras


class Brain(q_learning.Brain):
  def __init__(self):
    self.model_path = path.join(path.dirname(__file__), 'model')

  def build_model(self, learning_rate):
    self.model = keras.Sequential()

    self.model.add(keras.layers.InputLayer(input_shape=(38, 38, 3)))

    self.model.add(keras.layers.Convolution2D(32, kernel_size=(8, 8), strides=(4, 4)))
    self.model.add(keras.layers.Activation('relu'))

    self.model.add(keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='same'))

    self.model.add(keras.layers.Convolution2D(64, kernel_size=(4, 4), strides=(2, 2)))
    self.model.add(keras.layers.Activation('relu'))

    self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    self.model.add(keras.layers.Flatten())

    self.model.add(keras.layers.Dense(256))
    self.model.add(keras.layers.Activation('relu'))

    self.model.add(keras.layers.Dense(4))

    self.model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

  def save_model(self):
    self.model.save(self.model_path)

  def load_model(self):
    self.model = keras.models.load_model(self.model_path)

  def prepare_observation(self, observation):
    assert(observation.shape == (150, 150, 3))
    observation = cv2.resize(observation, dsize=(38, 38), interpolation=cv2.INTER_AREA)
    observation = observation / 255.0
    return observation

  def reset_prepare(self):
    pass
