from os import path

import cv2
import keras
from ml_tools import q_learning

from tensorflow import keras
from keras.applications.mobilenet_v3 import MobileNetV3Small


class Brain(q_learning.Brain):
  def __init__(self): 
    self.model_path = path.join(path.dirname(__file__), 'model')

    self.last_observations = None
    self.input_shape=(224, 224, 3)
    self.batched_input_shape = (None, *self.input_shape)

  def save_model(self):
    self.model.save(self.model_path)

  def load_model(self):
    self.model = keras.models.load_model(self.model_path)

  def build_model(self, learning_rate):
    input = keras.layers.Input(shape=self.input_shape)
    mobileNet = MobileNetV3Small(include_top=False, input_shape=self.input_shape)
    # mobileNet.trainable = False

    x = mobileNet(input, training=False)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(4, activation='relu')(x)

    self.model = keras.Model(inputs=input, outputs=output)
    self.model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

  def prepare_observation(self, observation):
    assert(observation.shape == (150, 150, 3))

    if self.last_observations is not None:
      alpha = 0.6
      beta = (1.0 - alpha)
      observation = cv2.addWeighted(observation, alpha, self.last_observations, beta, 0)
    self.last_observations = observation

    observation = cv2.resize(observation, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    # self.ui.video.update_stream('brain', observation)
    
    observation = observation / 255.0
    return observation

  def reset_prepare(self):
    self.last_observations = None
