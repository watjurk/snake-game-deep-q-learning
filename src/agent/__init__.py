from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class SnakeDirection(Enum):
    SNAKE_DIRECTION_UP = 0
    SNAKE_DIRECTION_RIGHT = 1
    SNAKE_DIRECTION_DOWN = 2
    SNAKE_DIRECTION_LEFT = 3


class Agent(ABC):
    @abstractmethod
    def get_action(self, observation: np.array) -> SnakeDirection:
        pass
