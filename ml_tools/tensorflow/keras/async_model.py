import multiprocessing
import os
import shutil
import tempfile
import threading

import tensorflow.keras

keras = tensorflow.keras

__all__ = ['makeAsyncModel']

def makeAsyncModel(model: keras.Model, async_prediction_model: keras.Model) -> keras.Model:
  asyncModel = AsyncModel(model, async_prediction_model)

  def fit(*args, **kwargs):
    asyncModel.fit(*args, **kwargs)

  model.fit = fit
  return model


class AsyncModel():
  def __init__(self, model: keras.Model, async_prediction_model: keras.Model):
    self.async_prediction_model = async_prediction_model

    temp_model_save_path = tempfile.mkdtemp(f"ml_tools-{os.getpid()}-AsyncFitModel")
    model.save(temp_model_save_path)

    self.fit_queue = multiprocessing.Queue()
    self.result_weights_queue = multiprocessing.Queue()

    multiprocessing.Process(
       target=AsyncModel._fit_process_start,
       args=(temp_model_save_path, self.fit_queue, self.result_weights_queue)
    ).start()

    threading.Thread(
      target=AsyncModel._join_weights_thread_start,
      args=(self.async_prediction_model, self.result_weights_queue)
    ).start()

  def fit(self, *args, **kwargs):
    self.fit_queue.put((args, kwargs))

  def _join_weights_thread_start(async_prediction_model: keras.Model, result_weights_queue: multiprocessing.Queue):
    while True:
      async_prediction_model.set_weights(result_weights_queue.get())

  def _fit_process_start(temp_model_save_path: str, fit_queue: multiprocessing.Queue, result_weights_queue: multiprocessing.Queue):
    model: keras.Model = keras.models.load_model(temp_model_save_path)
    shutil.rmtree(temp_model_save_path)

    while True:
      (args, kwargs) = fit_queue.get()
      model.fit(*args, **kwargs)
      result_weights_queue.put(model.get_weights())
