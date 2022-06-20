from os import path

import tensorflow.keras

keras = tensorflow.keras


class Brain():
  current_folder: str
  model: keras.Model

  def __init__(self) -> None:
    raise RuntimeError("default Brain not implemented.")

  def build_and_compile_model(self, **kwargs) -> None:
    model = self._build_model()
    self._compile_model(model, **kwargs)
    self.model = model

  def save_model(self, name: str, model_name: str) -> None:
    self._save_model(name, model_name, self.model)

  def load_model(self, name: str, model_name: str) -> None:
    self.model = self._load_model(name, model_name)

  def load_layers_and_compile_model(self, name: str, model_name: str, *, num_layers: int, trainable: bool = True, **kwargs) -> None:
    loaded_model = self._load_model(name, model_name)
    model = self._build_model()

    if num_layers > 0:
      print("Loaded layer:")
      print("=" * 65)
      for i in range(num_layers):
        print(model.layers[i].name)
        model.layers[i].set_weights(loaded_model.layers[i].get_weights())
        model.layers[i].trainable = trainable
      print("=" * 65)

    self._compile_model(model, **kwargs)
    self.model = model

  def _save_model(self, name: str, model_name: str, model: keras.Model) -> None:
    model.save(path.join(self.current_folder, name, model_name))

  def _load_model(self, name: str, model_name: str) -> keras.Model:
    return keras.models.load_model(path.join(self.current_folder, name, model_name))

  def _build_model(self) -> keras.Model:
    raise RuntimeError("default Brain not implemented.")

  def _compile_model(self, model: keras.Model, **kwargs) -> None:
    raise RuntimeError("default Brain not implemented.")
