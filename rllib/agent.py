import itertools
import random
from abc import ABC, abstractmethod

from gymnasium import Env
from rllib.model import DQN
from rllib.network import BasicNetwork
from tqdm.auto import tqdm

from rllib.buffer import PrioritizedReplayBuffer, ReplayBuffer
from rllib.utils import CsvWriter


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
        obs_shape = env.observation_space.shape
        self.lr = lr
        # if the network is not designated, use BasicNetwork
        if net is None:
            self.net = BasicNetwork(obs_shape[0], env.action_space.n)
        else:
            self.net = net
        if self.verbose == "debug":
            print("Network is Initialized")
            print(self.net)
        self.step = step
        self.batch_size = batch_size
        if prioritized_buffer:
            self.buffer = PrioritizedReplayBuffer(
                buffer_size, obs_shape, env.action_space.n, gamma
            )
        else:
            self.buffer = ReplayBuffer(
                buffer_size, obs_shape, env.action_space.n, gamma
            )
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
        self.algo = DQN(self.net, lr, device)

    def _chooseAction(self, obs):
        """Return Discrete actions"""
        print(obs)
        if random.random() < self.eps:
            return self.env.action_space.sample()
        return self.algo.choose_action(obs)

    def _optimize(self):
        if len(self.buffer) < self.batch_size:
            return
        transitions = self.buffer.sample(self.batch_size)
        self.algo.optimize(transitions)

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
        if log_path is not None:
            logger = CsvWriter("DQN", log_path)
        for episode in range(episodes):
            total_reward = 0
            total_loss = []
            obs = self.env.reset()
            done = False

            for t in tqdm(itertools.count()):
                self.env.render()
                act = self._chooseAction(obs)
                print("This is action:", act)
                obs_, reward, done, _, _ = self.env.step(act)
                total_reward += reward
                self.buffer.push(obs, act, reward, done, obs_)
                loss = self._optimize()
                total_loss.append(loss)
                if done:
                    break
                obs = obs_
            # update target model for every n episodes
            if (episode + 1) % self.step == 0:
                self.algo.update_target()
            avg_loss = sum(total_loss) / len(total_loss)
            if verbose:
                print(f"[{episode}] | Reward: {total_reward} | Avg Loss: {avg_loss}")
            if (save_every > 0) and (episode + 1) % save_every == 0:
                self.algo.save(f"./models/episodes_{episode}")
            if log_path is not None:
                logger.log(episode, total_reward, avg_loss)
