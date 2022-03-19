import signal

signal.signal(signal.SIGINT, lambda sig, frame: exit(0))

#

import time

import gym
import gym_snake
from ml_tools import q_learning
from ml_tools.ui import UI

ui = UI()
ui.connect("./public")

from brain.snake_v0.brain import Brain

brain = Brain()

action_space = [0, 1, 2, 3]
agent = q_learning.Agent(
  action_space=action_space,

  brain=brain,

  epsilon=1,
  epsilon_min=0.0001,
  epsilon_decay=0.00002,

  discount_factor=0.97,
  replay_batch_size=64,

  model_batch_size=32,
  model_learning_rate=0.0005,

  steps_to_train=10,
  steps_to_update_target=50,

  replay_memory_max_len=50_000,
  replay_memory_min_len=50_000 / 2,
)

env = gym.make('snake-v0')

raw_observation = env.reset()
agent.reset(raw_observation)

while True:
  ui.video.update_stream("test", raw_observation)

  speed = int(ui.control.get_value('speed') or 100)
  if speed != 100:
    time.sleep(1 / speed)

  action = agent.act()
  raw_next_observation, reward, done, info = env.step(action)
  agent.step(reward, raw_next_observation)

  raw_observation = raw_next_observation
  if done:
    raw_observation = env.reset()
    agent.reset(raw_observation)


