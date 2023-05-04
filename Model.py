import logging
from abc import ABC, abstractmethod

import torch
from torch.nn import Module
from torch.optim import Optimizer


class NetworkContainer(ABC):
    def __init__(self,
                 net: Module,
                 criterion: Module,
                 optimizer: Optimizer,
                 random_policy: None,
                 device: str):
        """
        net: A pytorch neural network object
        criterion: usually mse, but Huber loss is recommended
        optimizer: A pytorch optimizer
        """
        self.init_network()
        self.optimizer = optimizer
        self.criterion = criterion
        self.net = net
        self.random_policy = random_policy
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
        pass

    @abstractmethod
    def action_distribution(self, state):
        '''Returns the distribution of action'''
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError


class DQN(NetworkContainer):
    """
    This is an implementation of NatureDQN.
    It is used as a base model for DQN in this project.
    """
    def __init__(self,
                 net: Module,
                 criterion: Module,
                 optimizer: Optimizer,
                 device: str):
        super().__init__(net, criterion, optimizer, device)

    def update_target(self):
        """update target net parameters"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def init_network(self):
        self.policy_net = self.net
        self.target_net = self.net
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.update_target()

    def action_distribution(self, state):
        return self.policy_net(state)

    def optimize(self, transitions):
        # transfer data to GPU
        transitions = [torch.tensor(i, dtype=torch.float32, device=self.device) for i in transitions]
        states, actions, rewards, masks, states_ = transitions
        # calculate Q value
        q = self.action_distribution(states).gather(1, actions)
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
