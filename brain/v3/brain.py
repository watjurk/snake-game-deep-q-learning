from os import path

import tensorflow.keras
from ml_tools import q_learning

keras = tensorflow.keras

class Brain(q_learning.Brain):
  def __init__(self, learning_rate: float):
    self.current_folder = path.dirname(__file__)

    self.learning_rate = learning_rate

    self.input_shape = (50, 50, 3)
    self.batched_input_shape = (None, *self.input_shape)

  def _build_model(self):
    input = keras.layers.Input(shape=self.input_shape)

    x = keras.layers.Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu')(input)
    x = keras.layers.MaxPool2D(pool_size=(4, 4), strides=(2, 2), padding='same')(x)

    x = keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(256, activation='relu')(x)

    output = keras.layers.Dense(4, activation='linear')(x)

    return keras.Model(inputs=input, outputs=output)

  def _compile_model(self, model: keras.Model, **kwargs):
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), **kwargs)
