import cv2
import gym
import gym_snake
import numpy as np
import torch

SNAKE_WINDOW_NAME = "snake"
SNAKE_ACTION_UP = 0
SNAKE_ACTION_RIGHT = 1
SNAKE_ACTION_DOWN = 2
SNAKE_ACTION_LEFT = 3

WAIT_KEY_TIME_SECONDS = 2


cv2.namedWindow(SNAKE_WINDOW_NAME)

env = gym.make("snake-v0")
observation = env.reset()

while True:
    keyPressed = cv2.waitKey(WAIT_KEY_TIME_SECONDS * 1000) & 0xFF
    if keyPressed == ord("q"):
        break

    action = None
    if keyPressed == ord("w"):
        action = SNAKE_ACTION_UP

    if keyPressed == ord("s"):
        action = SNAKE_ACTION_DOWN

    if keyPressed == ord("a"):
        action = SNAKE_ACTION_LEFT

    if keyPressed == ord("d"):
        action = SNAKE_ACTION_RIGHT

    if action is None:
        continue

    (observation, reward, terminated, info) = env.step(action)
    cv2.imshow(SNAKE_WINDOW_NAME, observation)


cv2.destroyAllWindows()
