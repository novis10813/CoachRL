from abc import ABC, abstractmethod
from typing import Union
from itertools import count

import torch
import gymnasium as gym


class BaseTrainer(ABC):
    def __init__(
        self,
        env: gym.Env,
        lr: float,
        tau: float,
        batch_size: int,
        device: Union[str, None],
    ):
        self.env = env
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
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
    def _choose_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def _optimize(self):
        raise NotImplementedError

    def _update(self, policy_model, target_model, tau):
        for tar, pol in zip(target_model.parameters(), policy_model.parameters()):
            tar.data.copy_(tau * pol.data + (1 - tau) * tar.data)
        return

    # @abstractmethod
    # def record(self, **kwargs):
    #     raise NotImplementedError

    def train(self, episodes: int, **kwargs):
        for episode in range(episodes):
            observation, _ = self.env.reset()
            state = torch.tensor(observation, device=self.device).unsqueeze(0)
            done = False
            rewards = 0

            for t in count():
                self.env.render()
                action = self._choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(
                    action.item()
                )
                next_state = torch.tensor(next_state).unsqueeze(0)
                reward = torch.tensor(reward).unsqueeze(0)
                done = terminated or truncated

                # Store transition in buffer
                self.buffer.add(state, next_state, action, reward, done)

                # Go to next state
                state = next_state
                rewards += reward.item()

                loss = self._optimize()
                self._update(self.target_model, self.policy_model, self.tau)

                if done:
                    print(rewards)
                    # self.record(t, reward, loss, info)
                    break
