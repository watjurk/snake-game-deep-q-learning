import random

import numpy as np
import tensorflow.keras

keras = tensorflow.keras


def act(observation, model, epsilon, action_space):
  if random.random() <= epsilon:
    # explore
    action_index = random.randint(0, len(action_space) - 1)
    return (action_index, action_space[action_index])
  else:
    # exploit
    q_values = model.predict(np.array([observation]))
    action_index = np.argmax(q_values)
    return (action_index, action_space[action_index])


def train(model: keras.Model, target_model: keras.Model, model_batch_size, replay_memory, replay_batch_size, discount_factor):
  # See https://en.wikipedia.org/wiki/Q-learning
  learning_rate = 1
  replay_memory_batch = random.sample(replay_memory, min(replay_batch_size, len(replay_memory)))

  # observations (states)
  X = []

  # new q values
  Y = []

  observations = np.array([replay[0] for replay in replay_memory_batch])
  old_qs_values = model.predict(observations)

  next_observations = np.array([replay[3] for replay in replay_memory_batch])
  future_qs_values = target_model.predict(next_observations)
  future_max_qs = [max(future_q_values) for future_q_values in future_qs_values]

  for index, [observation, action_index, reward, next_observation] in enumerate(replay_memory_batch):
    old_q_values = old_qs_values[index]
    old_q = old_q_values[action_index]

    future_max_q = future_max_qs[index]

    new_q = old_q + (learning_rate * (reward + (discount_factor * future_max_q) - old_q))

    # Update desired q value.
    old_q_values[action_index] = new_q

    X.append(observation)
    Y.append(old_q_values)

  model.fit(np.array(X), np.array(Y), batch_size=model_batch_size)
