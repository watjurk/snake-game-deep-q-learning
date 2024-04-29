from typing import Optional

import cv2
import gym
import gym_snake
from agent import SnakeDirection
from agent.human import HumanAgent

SNAKE_WINDOW_NAME = "snake"
WAIT_KEY_TIME_SECONDS = 2


cv2.namedWindow(SNAKE_WINDOW_NAME)

# gym_snake registers the snake-v0 environment.
_ = gym_snake
env = gym.make("snake-v0")

observation = env.reset()
cv2.imshow(SNAKE_WINDOW_NAME, observation)

agent = HumanAgent()

while True:
    keyPressed = cv2.waitKey(WAIT_KEY_TIME_SECONDS * 1000) & 0xFF
    if keyPressed == ord("q"):
        break

    action: Optional[SnakeDirection] = None
    if isinstance(agent, HumanAgent):
        action = agent.get_action_from_keypress(keyPressed)

    if action is None:
        continue

    (observation, reward, terminated, info) = env.step(action.value)
    cv2.imshow(SNAKE_WINDOW_NAME, observation)


cv2.destroyAllWindows()
