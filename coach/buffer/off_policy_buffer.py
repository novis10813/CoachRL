import torch
import random
import numpy as np
import gymnasium as gym

from abc import ABC, abstractmethod
from typing import Union
from coach.buffer.utils import get_obs_shape, get_action_dim, get_device
from coach.buffer.sumtree import SumTree


class BaseReplayBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        gamma: float,
        device: Union[torch.device, str] = "auto",
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.gamma = gamma
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.observations = torch.empty(
            self.buffer_size, *self.obs_shape, dtype=torch.float32
        )
        self.actions = torch.empty(self.buffer_size, self.action_dim, dtype=torch.long)
        self.rewards = torch.empty(self.buffer_size, dtype=torch.float32)
        self.masks = torch.empty(self.buffer_size, dtype=torch.float32)

    def __len__(self):
        return self.buffer_size if self.full else self.pos

    @abstractmethod
    def add(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def sample(self, batch_size: int):
        raise NotImplementedError()


class ReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        gamma: float,
        device: Union[torch.device, str] = "auto",
    ):
        super().__init__(buffer_size, observation_space, action_space, gamma, device)

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        mask: torch.Tensor,
    ):
        self.observations[self.pos] = obs
        self.observations[(self.pos + 1) % self.buffer_size] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.masks[self.pos] = 0.0 if mask else self.gamma

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        if self.full:
            batch_idxs = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_idxs = np.random.randint(0, self.pos, size=batch_size)

        data = (
            self.observations[batch_idxs, :].to(self.device),
            self.actions[batch_idxs, :].to(self.device),
            self.observations[(batch_idxs + 1) % self.buffer_size, :].to(self.device),
            self.rewards[batch_idxs].to(self.device),
            self.masks[batch_idxs].to(self.device),
        )
        return data


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        gamma: float,
        device: Union[torch.device, str] = "auto",
        epsilon: float = 0.01,
        alpha: float = 0.1,
        beta: float = 0.1,
    ):
        super().__init__(buffer_size, observation_space, action_space, gamma, device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.tree = SumTree(self.buffer_size)
        self.max_priority = epsilon

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        mask: torch.Tensor,
    ):
        self.tree.add(self.max_priority, self.pos)
        super().add(obs, next_obs, action, reward, mask)

    def sample(self, batch_size: int):
        batch_idxs, tree_idxs = [], []
        priorities = np.empty((batch_size, 1))
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = random.uniform(a, b)
            tree_idx, priority, batch_idx = self.tree.get(cumsum)
            priorities[i] = priority
            tree_idxs.append(tree_idx)
            batch_idxs.append(batch_idx)
        probs = priorities / self.tree.total
        weights = (self.__len__() * probs) ** (-self.beta)
        weights /= weights.max()

        data = (
            self.observations[batch_idxs, :].to(self.device),
            self.actions[batch_idxs, :].to(self.device),
            self.observations[(batch_idxs + 1) % self.buffer_size, :].to(self.device),
            self.rewards[batch_idxs].to(self.device),
            self.masks[batch_idxs].to(self.device),
        )
        return data, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        """
        data_idxs: take tree_idxs from `sample`
        priorities: TD-error from model
        """
        for idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
