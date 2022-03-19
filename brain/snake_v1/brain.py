"""
Basic brain, without "frame stacking"
differences with previous:
- using gray scaling
- resizing size is 50, 50

"""

from os import path

import cv2
from ml_tools import general
from tensorflow import keras


class Brain(general.Brain):
  def __init__(self):
    self.model_path = path.join(path.dirname(__file__), 'model')

  def init(self, learning_rate):
    if not hasattr(self, 'model'):
      self.model = self._build_model(learning_rate)

  def save_model(self):
    self.model.save(self.model_path)

  def load_model(self):
    self.model = keras.models.load_model(self.model_path)

  def _build_model(self, learning_rate):
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(50, 50, 1)))

    model.add(keras.layers.Convolution2D(32, kernel_size=(8, 8), strides=(4, 4)))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='same'))

    model.add(keras.layers.Convolution2D(64, kernel_size=(4, 4), strides=(2, 2)))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(4))

    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    return model

  def prepare_observation(self, observation):
    assert(observation.shape == (150, 150, 3))

    observation = cv2.resize(observation, dsize=(50, 50), interpolation=cv2.INTER_AREA)
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    observation = observation.reshape(observation.shape + (1,))

    observation = observation / 255.0
    return observation

  def reset_prepare(self):
    pass
