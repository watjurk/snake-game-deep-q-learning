import collections
import json
import os

import tensorflow.keras
from ml_tools.tensorflow.keras.async_model import AsyncModel, makeAsyncModel

keras = tensorflow.keras

from . import algorithm
from .brain import Brain
from .preprocessor import Preprocessor


class Agent:
  def __init__(self,
               brain, preprocessor,
               action_space,
               epsilon_start, epsilon_stop, epsilon_decay_num_steps,
               discount_factor,
               replay_batch_size,
               model_batch_size,
               steps_to_train, steps_to_update_target,
               replay_memory_min_len, replay_memory_max_len,
               use_async_model=True
               ):

    self.params_dict = locals()

    # Delete non-representable params.
    del self.params_dict['self']
    del self.params_dict['brain']
    del self.params_dict['preprocessor']

    self.training = True

    self.brain: Brain = brain
    self.preprocessor: Preprocessor = preprocessor

    self.action_space = action_space

    self.epsilon = epsilon_start
    self.epsilon_stop = epsilon_stop
    self.epsilon_step = _smooth_transition_step(epsilon_start, epsilon_stop, epsilon_decay_num_steps)

    self.discount_factor = discount_factor

    self.replay_batch_size = replay_batch_size

    self.model_batch_size = model_batch_size

    self.steps_to_train = steps_to_train
    self.steps_to_update_target = steps_to_update_target

    self.steps_to_train_count = 0
    self.steps_to_update_target_count = 0

    # When replay_memory_min_len is reached traning is being considered.
    self.replay_memory_min_len = replay_memory_min_len
    self.replay_memory_max_len = replay_memory_max_len
    self.replay_memory = collections.deque(maxlen=self.replay_memory_max_len)

    if not hasattr(self.brain, 'model'):
      raise ValueError("Brain is expected to have model field")

    self.model = self.brain.model

    self.target_model = keras.models.clone_model(self.model)
    self.target_model.set_weights(self.model.get_weights())

    self.use_async_model = use_async_model
    if self.use_async_model:
      self.async_prediction_model = keras.models.clone_model(self.model)
      self.async_prediction_model.set_weights(self.model.get_weights())

      self.model = makeAsyncModel(self.model, self.async_prediction_model)
      self.brain.model = self.model

    self.observation = None
    self.action_index = None

  def reset(self, observation):
    self.preprocessor.reset()

    self.observation = self.preprocessor.preprocess(observation)
    self.action_index = None

    if self.epsilon > self.epsilon_stop:
      self.epsilon -= self.epsilon_step

  def act(self):
    if self.observation is None:
      raise ValueError("You are supposed to call reset once before act.")

    model = self.async_prediction_model if self.use_async_model else self.model
    action_index, action = algorithm.act(self.observation, model, self.epsilon, self.action_space)
    self.action_index = action_index

    return action

  def step(self, reward, next_observation):
    if self.action_index is None:
      raise ValueError("You are supposed to call act before step.")

    next_observation = self.preprocessor.preprocess(next_observation)

    # We need to preserve prepare_observation state and not corrupt replay_memory
    # because of that we return here.
    if not self.training:
      return

    self.replay_memory.append((self.observation, self.action_index, reward, next_observation))
    self.observation = next_observation

    if len(self.replay_memory) < self.replay_memory_min_len:
      return

    self._update_steps()

    if self.steps_to_train_count >= self.steps_to_train:
      algorithm.train(self.model, self.target_model,
                      self.model_batch_size,
                      self.replay_memory, self.replay_batch_size,
                      self.discount_factor)
      self.steps_to_train_count = 0

    if self.steps_to_update_target_count >= self.steps_to_update_target:
      self.target_model.set_weights(self.model.get_weights())
      self.steps_to_update_target_count = 0

  def _update_steps(self):
    self.steps_to_train_count += 1
    self.steps_to_update_target_count += 1

  def save_params(self, name: str) -> None:
    with open(os.path.join(name, 'agent_params.json'), 'w+') as outfile:
      json.dump(self.params_dict, outfile, indent=2, sort_keys=False)

def _smooth_transition_step(start, stop, num_steps):
  assert(start >= stop)

  step = (start / num_steps) - (stop / num_steps)
  return step
