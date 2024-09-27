from typing import Optional

import numpy as np

from agent import Agent, SnakeDirection


class HumanAgent(Agent):
    def get_action(self, observation: np.array) -> SnakeDirection:
        raise RuntimeError(
            "get_action called on HumanAgent. Use get_action_from_keypress."
        )

    def step(
        self,
        observation_s1: np.array,
        action_s1: SnakeDirection,
        reward_s2: float,
        observation_s2: np.array,
    ) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_action_from_keypress(self, keyPressed: int) -> Optional[SnakeDirection]:
        action = None
        if keyPressed == ord("w"):
            action = SnakeDirection.SNAKE_DIRECTION_UP

        if keyPressed == ord("s"):
            action = SnakeDirection.SNAKE_DIRECTION_DOWN

        if keyPressed == ord("a"):
            action = SnakeDirection.SNAKE_DIRECTION_LEFT

        if keyPressed == ord("d"):
            action = SnakeDirection.SNAKE_DIRECTION_RIGHT

        if action is None:
            return None

        return action
