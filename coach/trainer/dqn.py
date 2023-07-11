from coach.trainer.base import BaseTrainer
from coach.buffer.off_policy_buffer import ReplayBuffer, PrioritizedReplayBuffer

from typing import Union
import random
import copy
import gymnasium as gym
import torch
import torch.nn as nn


class DQN(BaseTrainer):
    def __init__(
        self,
        env: gym.Env,
        lr: float,
        tau: float,
        batch_size: int,
        buffer_size: int,
        gamma: float,
        prioritized: bool,
        eps: float,
        device: Union[str, None],
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.eps = eps
        self.prioritized = prioritized
        super().__init__(env, lr, tau, batch_size, device)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=self.lr, amsgrad=True
        )
        self.record = []

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

    def _optimize(self):
        if len(self.buffer) < self.batch_size:
            return

        obs, acts, next_obs, rewards, masks = self.buffer.sample(self.batch_size)
        Q_value = self.policy_model(obs).gather(1, acts)
        next_v = self.target_model(next_obs).max(1)[0].detach()
        next_Q_value = next_v * masks + rewards
        loss = self.criterion(Q_value, next_Q_value.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _choose_action(self, obs):
        if self.eps > random.random():
            return self.policy_model(obs).argmax()
        else:
            return torch.tensor(
                self.env.action_space.sample(), device=self.device, dtype=torch.long
            )

    # def record(self, t, reward, loss, info):
    #     durations_t = torch.tensor(self.record, dtype=torch.float)
    #     plt.clf()
    #     plt.title("Training")
    #     plt.xlabel("Episode")
    #     plt.ylabel("Duration")
    #     plt.plot(durations_t.numpy())
    #     if len(durations_t) >= 100:
    #         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #         means = torch.cat((torch.zeros(99), means))
    #         plt.plot(means.numpy())
    #     plt.pause(0.001)
    #     display.display(plt.gcf())
    #     display.clear_output(wait=True)
