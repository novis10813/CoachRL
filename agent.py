import itertools
from abc import ABC, abstractmethod

import torch
from gymnasium import Env
from tqdm.auto import tqdm

from buffers import BaseReplayBuffer
from Model import NetworkContainer


class BaseAgent(ABC):
    def __init__(self,
                 env: Env,
                 model: NetworkContainer,
                 buffer: BaseReplayBuffer,
                 args_config: dict):
        """
        args:
            env: A gymnasium base environment
            model: A NetWorkContainer object
            buffer: A ReplayBuffer object
            args_config: A dictionary object, contains training arguments
        """
        self.env = env
        self.model = model
        self.buffer = buffer
        # args_config
        self.episodes = args_config["episodes"]
        self.batch_size = args_config["batch_size"]
        self.record = args_config["record"]
        self.mode = args_config["action_mode"]
        self.update_step = args_config["update_step"]
    
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
    def test(self, env):
        raise NotImplementedError


class Agent(BaseAgent):
    def __init__(self,
                 env: Env,
                 model: NetworkContainer,
                 buffer: BaseReplayBuffer,
                 args_config: dict):
        super().__init__(env, model, buffer, args_config)

    def _chooseAction(self, obs):
        # TODO: make it more straightforward(put it in config or what?)
        assert self.mode in ["eps", "boltz", None], "Not an available method"
        with torch.no_grad():
            action = self.model.action_distribution(obs).max(1)[1].view(1, 1)

        if self.mode is None:
            return action
        if self.mode == "eps":
            pass
        if self.mode == "boltzman":
            pass

    def _optimize(self):
        if len(self.buffer) < self.batch_size:
            return
        transitions = self.buffer.sample(self.batch_size)
        self.model.optimize(transitions)

    def train(self, logger):
        for episode in range(self.episodes):
            total_reward = 0
            total_loss = []
            obs = self.env.reset()
            done = False

            for t in tqdm(itertools.count()):
                self.env.render()
                act = self._chooseAction(obs)
                obs_, reward, done, _, _ = self.env.step(act)
                total_reward += reward
                self.buffer.push(obs, act, reward, done, obs_)
                loss = self._optimize()
                total_loss.append(loss)
                if done:
                    break
                obs = obs_
            # update target model for every n steps
            if (episode+1) % self.update_step == 0:
                self.model.update_target()
            # TODO: show loss and reward every episodes
            avg_loss = sum(total_loss) / len(total_loss)
            print(f"[{episode}] | Reward: {total_reward} | Avg Loss: {avg_loss}")
            logger.log(episode, total_reward, avg_loss)
        logger.close()
