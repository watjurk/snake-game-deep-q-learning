import os
import random
from collections import deque

import numpy as np
import torch
from torch import nn

from agent import Agent, SnakeDirection


class DeepQAgent(Agent):
    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
        replay_memory_length: int,
        train_interval: int,
        train_batch_size: int,
        target_model_update_interval: int,
        observation_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.train_interval = train_interval
        self.train_batch_size = train_batch_size

        self.target_model_update_interval = target_model_update_interval

        model_observation_shape = (observation_shape[2], observation_shape[0], observation_shape[1])
        self.target_model = Model(model_observation_shape)
        self.model = Model(model_observation_shape)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.current_step = 0
        self.replay_memory = deque(maxlen=replay_memory_length)

    def get_action(self, observation: np.array) -> SnakeDirection:
        if 0.5 < random.random():
            return random.sample(
                [SnakeDirection.SNAKE_DIRECTION_UP, SnakeDirection.SNAKE_DIRECTION_RIGHT, SnakeDirection.SNAKE_DIRECTION_DOWN, SnakeDirection.SNAKE_DIRECTION_LEFT],
                k=1,
            )[0]

        BATCH_SIZE = 1
        observation = observation.reshape((BATCH_SIZE, 3, 150, 150))
        q_values: torch.Tensor = self.model(torch.tensor(observation, dtype=torch.float))
        q_values = q_values.detach().numpy()
        direction = np.argmax(q_values)
        match direction:
            case 0:
                return SnakeDirection.SNAKE_DIRECTION_UP
            case 1:
                return SnakeDirection.SNAKE_DIRECTION_RIGHT
            case 2:
                return SnakeDirection.SNAKE_DIRECTION_DOWN
            case 3:
                return SnakeDirection.SNAKE_DIRECTION_LEFT

    def step(
        self,
        observation_s1: np.array,
        action_s1: SnakeDirection,
        reward_s1_s2: float,
        observation_s2: np.array,
    ) -> None:
        self.current_step += 1
        self.replay_memory.append((observation_s1.reshape((3, 150, 150)), action_s1.value, reward_s1_s2, observation_s2.reshape((3, 150, 150))))

        if self.current_step % self.train_interval == 0:
            self._train()

        if self.current_step % self.target_model_update_interval == 0:
            self._update_target_model()

    def reset(self) -> None:
        pass

    def _train(self):
        if len(self.replay_memory) < self.train_batch_size:
            return

        replay_memory_sample = random.sample(self.replay_memory, k=self.train_batch_size)

        # observation_s1, action_s1, reward_s1_s2, observation_s2
        observations_s1 = [s[0] for s in replay_memory_sample]
        observations_s2 = [s[3] for s in replay_memory_sample]

        current_qs = self.target_model(torch.tensor(observations_s1, dtype=torch.float))
        future_qs: torch.Tensor = self.model(torch.tensor(observations_s2, dtype=torch.float))

        new_qs = current_qs.detach().clone()

        for i, (_, action_s1, reward_s1_s2, _) in enumerate(replay_memory_sample):
            current_q = current_qs[i]
            future_q = future_qs[i]

            new_qs[i][action_s1] = (1 - self.learning_rate) * current_q[action_s1] + self.learning_rate * (reward_s1_s2 * future_q.max())

        self.optimizer.zero_grad()

        inputs = torch.tensor(observations_s1, dtype=torch.float)
        labels = new_qs

        outputs = self.model(inputs)

        loss = torch.nn.functional.mse_loss(outputs, labels)

        loss.backward()
        self.optimizer.step()

    def _update_target_model(self):
        model_state_dict = self.model.state_dict()
        self.target_model.load_state_dict(model_state_dict)


class Model(nn.Module):
    EXPECTED_OBSERVATION_SHAPE = (3, 150, 150)

    def __init__(self, observation_shape) -> None:
        super().__init__()

        assert observation_shape == self.EXPECTED_OBSERVATION_SHAPE

        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(4, 4), stride=1, padding=2),
            nn.MaxPool2d(kernel_size=(3, 3)),
            #
            nn.Conv2d(96, 192, kernel_size=(3, 3), stride=1, padding=2),
            nn.MaxPool2d(kernel_size=(3, 3)),
            #
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=1, padding=2),
            nn.MaxPool2d(kernel_size=(3, 3)),
            #
            nn.Conv2d(192, 96, kernel_size=(3, 3), stride=1, padding=2),
            nn.Flatten(),
        )

        self.decision_tree = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6144, 1536),
            #
            nn.ReLU(),
            nn.Linear(1536, 512),
            #
            nn.ReLU(),
            nn.Linear(512, 256),
            #
            nn.ReLU(),
            nn.Linear(256, 32),
            #
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, observation: torch.Tensor):
        return self.decision_tree(self.convolutional(observation))


if __name__ == "__main__":
    BATCH_SIZE = 1

    m = Model((3, 150, 150))
    random_data = torch.rand((BATCH_SIZE, 3, 150, 150))
    out = m.convolutional(random_data)
    print(out.shape)
