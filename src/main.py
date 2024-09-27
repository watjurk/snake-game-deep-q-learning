from typing import Optional

import cv2
import gym

import gym_snake
from agent import SnakeDirection
from agent.deepq import DeepQAgent
from agent.human import HumanAgent

DEBUG = False  # single step the game
HUMAN = False

WAIT_FOR_KEYPRESS = False
if DEBUG:
    WAIT_FOR_KEYPRESS_SECONDS = True

SNAKE_WINDOW_NAME = "snake"
GRID_SIZE = (15, 15)
UNIT_SIZE = 10

EXPECTED_OBSERVATION_SHAPE = (GRID_SIZE[0] * UNIT_SIZE, GRID_SIZE[1] * UNIT_SIZE, 3)

cv2.namedWindow(SNAKE_WINDOW_NAME)

_ = gym_snake
env = gym.make(
    "snake-v0",
    grid_size=list(GRID_SIZE),
    unit_size=UNIT_SIZE,
)

observation = env.reset()
assert observation.shape == EXPECTED_OBSERVATION_SHAPE

cv2.imshow(SNAKE_WINDOW_NAME, observation)

agent = DeepQAgent(
    observation_shape=EXPECTED_OBSERVATION_SHAPE,
    learning_rate=0.001,
    discount_factor=0.01,
    replay_memory_length=1000,
    train_interval=32,
    train_batch_size=32,
    # train_batch_size=30,
    target_model_update_interval=64,
)

if HUMAN:
    agent = HumanAgent()


# cv2 specific behaviour
cv2_wait_time = 1
if WAIT_FOR_KEYPRESS:
    cv2_wait_time = 0


while True:
    keyPressed = cv2.waitKey(cv2_wait_time) & 0xFF
    if keyPressed == ord("q"):
        break

    action: Optional[SnakeDirection] = None
    if isinstance(agent, HumanAgent):
        action = agent.get_action_from_keypress(keyPressed)
    else:
        action = agent.get_action(observation)

    if action is None:
        continue

    (next_observation, reward, terminated, _info) = env.step(action.value)
    assert next_observation.shape == EXPECTED_OBSERVATION_SHAPE

    if terminated:
        agent.reset()
        observation = env.reset()
        continue

    agent.step(
        observation_s1=observation,
        action_s1=action,
        reward_s1_s2=reward,
        observation_s2=next_observation,
    )

    observation = next_observation
    cv2.imshow(SNAKE_WINDOW_NAME, observation)


cv2.destroyAllWindows()
