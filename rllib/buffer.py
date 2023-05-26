import random
from abc import ABC, abstractmethod

import numpy as np

from rllib.sumtree import SumTree


class BaseReplayBuffer(ABC):
    def __init__(self, buffer_size: int, gamma: float):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.cursor = -1
        self.full = False

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.cursor + 1

    @abstractmethod
    def push(self, state, action, reward, done, next_state):
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size, replace=False, **kwargs):
        raise NotImplementedError


class ReplayBuffer(BaseReplayBuffer):
    """
    A basic ReplayBuffer
    Args:
        buffer_size: Maximum number of transitions a buffer can store
        obs_shape: The shape of observation_space
        action_shape: The shape of action_space, number of discrete actions
        gamma: The gamma in bellman equation, which is used to pre-calculate
            the value of (1 - done) * gamma.
            If `done`, the mask will record `0.0`, else `gamma`
    """

    def __init__(self, buffer_size, obs_size, action_size, gamma):
        super().__init__(buffer_size, gamma)
        self.state = np.zeros([buffer_size, obs_size])
        self.action = np.zeros([buffer_size, action_size])
        self.reward = np.zeros((buffer_size))
        self.mask = np.zeros((buffer_size))
        self.next_state = np.zeros([buffer_size, obs_size])

    def push(self, state, action, reward, done, next_state):
        # calculate cursor position and length
        self.cursor = (self.cursor + 1) % self.buffer_size
        # push data into the buffer
        self.state[self.cursor] = state
        self.action[self.cursor] = action
        self.reward[self.cursor] = reward
        # pre-calculate (1-done) * gamma
        self.mask[self.cursor] = 0.0 if done else self.gamma
        self.next_state[self.cursor] = next_state
        # check is full or not
        if self.cursor + 1 == self.buffer_size:
            self.full = True

    def sample(self, batch_size, replace=False):
        assert (
            batch_size <= self.__len__()
        ), "sampling size should smaller than current buffer size"
        indices = np.random.choice(self.__len__(), batch_size, replace=replace)
        return (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.mask[indices],
            self.next_state[indices],
        )


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self, buffer_size, obs_size, action_size, gamma, alpha=0.1, beta=0.1, eps=1e-2
    ):
        super().__init__(buffer_size, obs_size, action_size, gamma)
        self.tree = SumTree(buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps  # prevent zero probability
        self.max_priority = eps

    def push(self, state, action, reward, done, next_state):
        self.cursor = (self.cursor + 1) % self.buffer_size
        self.tree.add(self.max_priority, self.cursor)
        # push data into the buffer
        self.state[self.cursor] = state
        self.action[self.cursor] = action
        self.reward[self.cursor] = reward
        # pre-calculate (1-done) * gamma
        self.mask[self.cursor] = 0.0 if done else self.gamma
        self.next_state[self.cursor] = next_state
        if self.cursor + 1 == self.buffer_size:
            self.full = True

    def sample(self, batch_size):
        assert (
            batch_size <= self.__len__()
        ), "sampling size should smaller than current buffer size"
        sample_idxs, tree_idxs = [], []
        priorities = np.empty((batch_size, 1))
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = random.uniform(a, b)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)
            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)
        probs = priorities / self.tree.total
        weights = (self.__len__() * probs) ** -self.beta
        weights /= weights.max()
        batch = (
            self.state[sample_idxs],
            self.action[sample_idxs],
            self.reward[sample_idxs],
            self.mask[sample_idxs],
            self.next_state[sample_idxs],
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
