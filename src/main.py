import cv2
import numpy as np
import torch

print("Hello world!")
print("Torch version:", torch.__version__)

print("Torch default device:", torch.get_default_device())
print("Torch mps available:", torch.backends.mps.is_available())
print("Torch cuda available:", torch.cuda.is_available())


def nothing(x):
    pass


# Create a black image, a window
img = np.zeros((200, 200, 3), np.uint8)
cv2.namedWindow("image")

cv2.createTrackbar("R", "image", 0, 255, nothing)
cv2.createTrackbar("G", "image", 0, 255, nothing)
cv2.createTrackbar("B", "image", 0, 255, nothing)


while True:
    cv2.imshow("image", img)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    r = cv2.getTrackbarPos("R", "image")
    g = cv2.getTrackbarPos("G", "image")
    b = cv2.getTrackbarPos("B", "image")

    print(r, g, b)

    img[:] = [b, g, r]

cv2.destroyAllWindows()

import gym
import gym_snake

# Construct Environment
env = gym.make("snake-v0")
observation = env.reset()  # Constructs an instance of the game

# Controller
game_controller = env.controller

# Grid
grid_object = game_controller.grid
grid_pixels = grid_object.grid

# Snake(s)
snakes_array = game_controller.snakes
snake_object1 = snakes_array[0]
