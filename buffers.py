from abc import ABC, abstractmethod

import numpy as np


class BaseReplayBuffer(ABC):
    def __init__(self, buffer_size: int, gamma: float):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.cursor = -1
        self.full = False

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.cursor+1

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
        action_shape: The shape of action_space
        gamma: The gamma in bellman equation, which is used to pre-calculate
            the value of (1 - done) * gamma.
            If `done`, the mask will record `0.0`, else `gamma`
    """
    def __init__(self, buffer_size, obs_shape, action_shape, gamma):
        super().__init__(buffer_size, gamma)
        obs_shape = list(obs_shape)
        obs_shape.insert(0, buffer_size)
        self.state = np.zeros(obs_shape)
        action_shape = list(action_shape)
        action_shape.insert(0, buffer_size)
        self.action = np.zeros(action_shape)
        self.reward = np.zeros((buffer_size))
        self.mask = np.zeros((buffer_size))
        self.next_state = np.zeros(obs_shape)

    def push(self, state, action, reward, done, next_state):
        # calculate cursor position and length
        self.cursor = (self.cursor+1) % self.buffer_size
        # push data into the buffer
        self.state[self.cursor] = state
        self.action[self.cursor] = action
        self.reward[self.cursor] = reward
        # pre-calculate (1-done) * gamma
        self.mask[self.cursor] = 0.0 if done else self.gamma
        self.next_state[self.cursor] = next_state
        # check is full or not
        if self.cursor+1 == self.buffer_size:
            self.full = True

    def sample(self, batch_size, replace=False):
        assert batch_size <= self.__len__(), "sampling size should smaller than current buffer size"
        indices = np.random.choice(self.__len__(), batch_size, replace=replace)
        return (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.mask[indices],
            self.state[indices],
        )
