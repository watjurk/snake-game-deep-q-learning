import signal

signal.signal(signal.SIGINT, lambda sig, frame: exit(0))

#


def replica_fn(agent, replica_num, live):
  import time

  import gym
  import gym_snake

  # initialize
  env = gym.make('snake-v0')

  raw_observation = env.reset()
  agent.reset(raw_observation)

  speed = .001

  def on_speed(value):
    nonlocal speed
    speed = int(value) / 1000


  live.value.on("speed", on_speed)

  # run loop
  while True:
    if speed != 0:
      # live.stream.set_wait_time(speed)
      live.stream.update("raw_" + str(replica_num), raw_observation)
      time.sleep(speed)

    action = agent.act()
    raw_next_observation, reward, done, info = env.step(action)
    agent.step(reward, raw_next_observation)

    raw_observation = raw_next_observation
    if done:
      raw_observation = env.reset()
      agent.reset(raw_observation)


if __name__ == '__main__':
  import os

  from ml_tools import q_learning
  from ml_tools.live import Live

  from brain.snake_v0.brain import Brain

  brain = Brain()

  action_space = [0, 1, 2, 3]
  manager = q_learning.MultiAgentManager(
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

  live = Live(os.path.join(os.getcwd(), 'public'))

  manager.set_replica_fn(replica_fn)

  manager.run_replica(8, args=(live,))

  manager.wait()
