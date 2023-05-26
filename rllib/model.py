import os
from abc import ABC, abstractmethod

import torch
from torch.nn import Module, SmoothL1Loss
from torch.optim import AdamW


class BasicAlgo(ABC):
    def __init__(self, net: Module, lr: float, tau: float, device: str):
        """
        net: A pytorch neural network object
        optimizer: A pytorch optimizer
        """
        self.optimizer = AdamW(net.parameters(), lr=lr)
        self.criterion = SmoothL1Loss()
        self.net = net
        self.tau = tau
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    @abstractmethod
    def init_network(self):
        """Initialize the model"""
        raise NotImplementedError

    @abstractmethod
    def optimize(self, transitions):
        """Updates the model"""
        raise NotImplementedError

    @abstractmethod
    def soft_update(self):
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError


class DQN(BasicAlgo):
    """
    This is an implementation of NatureDQN.
    It is used as a base model for DQN in this project.
    """

    def __init__(self, net: Module, lr: float, tau: float, device: str):
        super().__init__(net, lr, tau, device)
        self.init_network()

    def soft_update(self):
        """update target net parameters"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict.keys():
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * 0.005 + target_net_state_dict[key] * (1 - 0.005)
        self.target_net.load_state_dict(target_net_state_dict)

    def hard_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def init_network(self):
        self.policy_net = self.net.to(self.device)
        self.target_net = self.net.to(self.device)
        self.hard_update()

    def save(self, model_path="./model"):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(
            self.policy_net.state_dict(), os.path.join(model_path, "policy.ckpt")
        )
        torch.save(
            self.target_net.state_dict(), os.path.join(model_path, "target.ckpt")
        )

    def load(self, model_path="./model"):
        self.policy_net.load_state_dict(
            torch.load(os.path.join(model_path, "policy.ckpt"))
        )
        self.target_net.load_state_dict(
            torch.load(os.path.join(model_path, "target.ckpt"))
        )

    def choose_action(self, state):
        with torch.no_grad():
            act = self.policy_net(state)
            return torch.argmax(act).cpu().numpy().item()

    def optimize(self, transitions):
        # transfer data to GPU
        states = torch.tensor(transitions[0], dtype=torch.float32, device=self.device)
        actions = torch.tensor(transitions[1], dtype=torch.long, device=self.device)
        rewards = torch.tensor(transitions[2], dtype=torch.float32, device=self.device)
        masks = torch.tensor(transitions[3], dtype=torch.float32, device=self.device)
        states_ = torch.tensor(transitions[4], dtype=torch.float32, device=self.device)
        # calculate Q value
        q = self.policy_net(states).gather(1, actions)
        # calculate expected Q value
        with torch.no_grad():
            v_ = self.target_net(states_).max(1)[0]
        q_ = rewards + masks * v_
        # the buffer already deal with the mask,
        # so it don't have to filter those next_states which are done

        # compute Huber loss
        loss = self.criterion(q, q_.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        # for recording the loss
        return loss.item()
