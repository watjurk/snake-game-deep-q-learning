from os import path

import tensorflow.keras
from ml_tools import q_learning

keras = tensorflow.keras

class Brain(q_learning.Brain):
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate

    self.current_folder = path.dirname(__file__)
    self.model_folder_path = path.join(self.current_folder, 'model')

    self.last_observations = None
    self.input_shape = (50, 50, 3)
    self.batched_input_shape = (None, *self.input_shape)

  def build_and_compile_model(self, **kwargs):
    model = self._build_model()
    self._compile_model(model, **kwargs)
    self.model = model

  def save_model(self, name='trained'):
    self.model.save(path.join(self.model_folder_path, name))

  def load_layers_and_compile_model(self, *, name='trained', num_layers, **kwargs):
    loaded_model = self._load_model(name)
    model = self._build_model()
  
    if num_layers > 0:
      print("Loaded layer")
      print("=" * 65)
      for i in range(num_layers):
        print(model.layers[i].name)
        model.layers[i].set_weights(loaded_model.layers[i].get_weights())
      print("=" * 65)

    self._compile_model(model, **kwargs)
    self.model = model

  def load_model(self, name='trained'):
    self.model = self._load_model(name)

  def _load_model(self, name):
    return keras.models.load_model(path.join(self.model_folder_path, name))

  def _build_model(self):
    input = keras.layers.Input(shape=self.input_shape)

    x = keras.layers.Conv2D(128, kernel_size=(8, 8), activation='relu')(input)
    x = keras.layers.MaxPool2D(pool_size=(4, 4))(x)

    x = keras.layers.Conv2D(64, kernel_size=(4, 4), activation='relu')(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu')(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dense(16, activation='relu')(x)

    output = keras.layers.Dense(4, activation='softmax')(x)

    return keras.Model(inputs=input, outputs=output)

  def _compile_model(self, model: keras.Model, **kwargs):
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), **kwargs)
