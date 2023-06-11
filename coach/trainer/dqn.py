from coach.trainer.base import BaseTrainer
from coach.buffer.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

from typing import Union

import copy
import gymnasium as gym
import torch
import torch.nn as nn


class DQN(BaseTrainer):
    def __init__(
        self,
        env: gym.Env,
        lr: float,
        buffer_size: int,
        gamma: float,
        prioritized: bool,
        device: Union[str, None],
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.prioritized = prioritized
        super().__init__(env, lr, device)

    def _init_model(self):
        self.policy_model = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.env.action_space.n),
        )
        self.target_model = copy.deepcopy(self.policy_model)
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def _init_buffer(self):
        if self.prioritized:
            self.buffer = PrioritizedReplayBuffer(
                self.buffer_size,
                self.env.observation_space,
                self.env.action_space,
                self.gamma,
                self.device,
            )
        else:
            self.buffer = ReplayBuffer(
                self.buffer_size,
                self.env.observation_space,
                self.env.action_space,
                self.gamma,
                self.device,
            )
