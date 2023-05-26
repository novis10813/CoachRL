import itertools
import random
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from gymnasium import Env
from rllib.model import DQN
from rllib.network import BasicNetwork

from rllib.buffer import PrioritizedReplayBuffer, ReplayBuffer
from rllib.utils import CsvWriter, StateProcessor, Plotter

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


class BaseAgent(ABC):
    def __init__(
        self,
        env: Env,
        lr=0.0001,
        gamma=0.9,
        net=None,
        prioritized_buffer=False,
        buffer_size=100000,
        step=3,
        batch_size=512,
        verbose=False,
    ):
        assert verbose in [True, False, "debug"]
        self.verbose = verbose
        self.env = env
        action_size = 2
        obs_size = env.observation_space.shape[0]
        self.lr = lr
        self.buffer_size = buffer_size
        # if the network is not designated, use BasicNetwork
        if net is None:
            self.net = BasicNetwork(obs_size, action_size)
        else:
            self.net = net
        if self.verbose == "debug":
            print("Network is Initialized")
            print(self.net)
        self.step = step
        self.batch_size = batch_size
        if prioritized_buffer:
            self.buffer = PrioritizedReplayBuffer(
                buffer_size, obs_size, action_size, gamma
            )
        else:
            self.buffer = ReplayBuffer(buffer_size, obs_size, 1, gamma)
        if self.verbose == "debug":
            print("Buffer is Initialized")

    @abstractmethod
    def _optimize(self):
        raise NotImplementedError

    @abstractmethod
    def _chooseAction(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, obs):
        raise NotImplementedError

    @abstractmethod
    def load(self, model_path):
        raise NotImplementedError


class DQNAgent(BaseAgent):
    def __init__(
        self,
        env: Env,
        lr=0.0001,
        gamma=0.9,
        net=None,
        prioritized_buffer=False,
        buffer_size=100000,
        step=3,
        batch_size=512,
        verbose=False,
        eps=0.1,
        tau=0.005,
        device=None,
    ):
        """
        Paremeters
        ----------
        env : Env
            A gymnasium base environment.

        lr: float, default=0.0001
            Learning rate

        gamma : float, default=0.9
            Discount factor

        net : nn.Module, default=None
            A neural network based on PyTorch, if `None`, use BasicNetwork instead.

        prioritized_buffer : bool, default=False
            Whether to use prioritized replay buffer

        buffer_size : int, default=100000
            Size of the replay buffer

        step : int, default=3
            How many episodes per update for a target network

        batch_size : int, default=512
            When optimizing the network,
            the number of transitions sample from the buffer.

        eps : float, default=0.1
            Epsilon for epsilon-greedy action selection

        device : str or None, default=None
            Device to run the model.
            Use None to let PyTorch decide the device.
            {"cuda" or "cpu"} to decide on your own.
        """
        super().__init__(
            env,
            lr,
            gamma,
            net,
            prioritized_buffer,
            buffer_size,
            step,
            batch_size,
            verbose,
        )
        self.eps = eps
        self.algo = DQN(self.net, lr, tau, device)
        self.device = device
        self.steps = 0
        self.plot = Plotter()

    def _chooseAction(self, obs):
        """Return Discrete actions"""
        eps_threshold = 0.005 + (self.eps - 0.005) * math.exp(-1.0 * self.steps / 1000)
        self.steps += 1

        if random.random() <= eps_threshold:
            act = self.env.action_space.sample()
            return act

        act = self.algo.choose_action(obs)
        return act

    def _optimize(self):
        if len(self.buffer) < self.batch_size:
            return 0
        transitions = self.buffer.sample(self.batch_size)
        return self.algo.optimize(transitions)

    def load(self, model_path):
        self.algo.load(model_path)

    def predict(self, obs):
        return self.algo.choose_action(obs)

    def train(self, episodes=200, save_every=0, log_path=None, verbose=True):
        """
        Train the agent

        Parameters
        ----------
        episodes : int, default=200
            Number of training episodes

        save_every : int, default=0
            How often to save the model,
            if 0, only save for the last episode

        log_path : str or None, default=None
            Use str to designate the log path.
            default format is `datetime.now()`-DQN.csv
            If None, do not log.

        verbose : bool, default=True
            Whether to show loss and reward on terminal
        """
        processor = StateProcessor(self.device)
        # if log_path is not None:
        # logger = CsvWriter("DQN", log_path)
        for episode in range(episodes):
            # total_reward = 0
            # total_loss = []
            obs, _ = self.env.reset()
            obs = processor.to_tensor(obs)
            done = False

            for t in itertools.count():
                self.env.render()
                act = self._chooseAction(obs)
                obs_, reward, terminated, truncated, _ = self.env.step(act)
                done = terminated or truncated

                # Store transition
                self.buffer.push(processor.to_numpy(obs), act, reward, done, obs_)

                # Go to next state
                obs = processor.to_tensor(obs_)

                # Optimize policy model
                self._optimize()

                # Operate soft update
                self.algo.soft_update()

                # record loss and reward
                # total_reward += reward
                # total_loss.append(loss)
                if done:
                    self.plot.episodes_durations.append(t + 1)
                    self.plot.plot_durations()
                    break

            # avg_loss = sum(total_loss) / len(total_loss)
            # print(f"[{episode+1}] | Reward: {total_reward} Avg Loss: {avg_loss}")
            # avg_loss = 0
            # if (save_every > 0) and (episode + 1) % save_every == 0:
            # self.algo.save(f"./models/episodes_{episode}")
            # if log_path is not None:
            #     logger.log(episode, total_reward, avg_loss)
        self.plot.plot_durations(show_result=True)
        plt.ioff()
        plt.show()
