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

    @abstractmethod
    def step(
        self,
        observation_s1: np.array,
        action_s1: SnakeDirection,
        reward_s2: float,
        observation_s2: np.array,
    ) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset is called when the environment reaches the terminal state.
        """
        pass
