import os
from abc import ABC, abstractmethod

import torch
from torch.nn import Module, SmoothL1Loss
from torch.optim import AdamW


class BasicAlgo(ABC):
    def __init__(self,
                 net: Module,
                 lr: float,
                 device: str):
        """
        net: A pytorch neural network object
        criterion: usually mse, but Huber loss is recommended
        optimizer: A pytorch optimizer
        """
        self.optimizer = AdamW(net.parameters(), lr=lr)
        self.criterion = SmoothL1Loss()
        self.net = net
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
    
    @abstractmethod
    def init_network(self):
        '''Initialize the model'''
        raise NotImplementedError

    @abstractmethod
    def optimize(self, transitions):
        '''Updates the model'''
        raise NotImplementedError

    @abstractmethod
    def update_target(self):
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError


class DQN(BasicAlgo):
    """
    This is an implementation of NatureDQN.
    It is used as a base model for DQN in this project.
    """
    def __init__(self,
                 net: Module,
                 lr: float,
                 device: str):
        super().__init__(net, lr, device)
        self.init_network()

    def update_target(self):
        """update target net parameters"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def init_network(self):
        self.policy_net = self.net
        self.target_net = self.net
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.update_target()

    def save(self, model_path="./model"):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.policy_net.state_dict(), os.path.join(model_path, "policy.ckpt"))
        torch.save(self.target_net.state_dict(), os.path.join(model_path, "target.ckpt"))

    def load(self, model_path="./model"):
        self.policy_net.load_state_dict(torch.load(os.path.join(model_path, "policy.ckpt")))
        self.target_net.load_state_dict(torch.load(os.path.join(model_path, "target.ckpt")))

    def choose_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def optimize(self, transitions):
        # transfer data to GPU
        transitions = [torch.tensor(i, dtype=torch.float32, device=self.device) for i in transitions]
        states, actions, rewards, masks, states_ = transitions
        # calculate Q value
        q = self.policy_net(states).gather(1, actions)
        # calculate expected Q value
        with torch.no_grad():
            v_ = self.target_net(states_).max(1)[0]
        q_ = rewards + masks * v_
        # the buffer already deal with the mask,
        # so I don't have to filter those next_states which are done

        # compute Huber loss
        loss = self.criterion(q, q_.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        # for recording the loss
        return loss.item()
