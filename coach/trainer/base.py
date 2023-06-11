from abc import ABC, abstractmethod
from typing import Union
from itertools import count

import torch
import gymnasium as gym


class BaseTrainer(ABC):
    def __init__(self, env: gym.Env, lr: float, device: Union[str, None]):
        self.env = env
        self.lr = lr
        self.device = device
        self._init_model()
        self._init_buffer()

    @abstractmethod
    def _init_model(self):
        raise NotImplementedError

    @abstractmethod
    def _init_buffer(self):
        raise NotImplementedError

    # def load_model(self, **kwargs):
    #     raise NotImplementedError
    #
    # def save_model(self, **kwargs):
    #     raise NotImplementedError

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def optimize(self):
        raise NotImplementedError

    @abstractmethod
    def record(self, **kwargs):
        raise NotImplementedError

    def train(self, episodes: int, **kwargs):
        for episode in range(episodes):
            observation, info = self.env.reset()
            state = torch.tensor(observation, dtype=torch.float32, device=self.device)
            done = False

            for t in count():
                self.env.render()
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store transition in buffer
                self.buffer.add(state, action, reward, next_state, done)

                # Go to next state
                state = torch.tensor(
                    next_state, dtype=torch.float32, device=self.device
                )

                loss = self.optimize()

                if done:
                    self.record(t, reward, loss, info)
                    break
