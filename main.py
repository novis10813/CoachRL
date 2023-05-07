import gymnasium as gym
import torch.nn as nn
from torch.optim import AdamW

from agent import OffPolicyAgent
from buffers import ReplayBuffer
from Model import DQN
from wrappers import env_wrapper

env =gym.make("ALE/Asteroids-v5", render_mode="human", obs_type="ram")
# env = env_wrapper(env)
env.metadata['render_fps'] = 15
model = DQN(net=, optimizer=, criterion=nn.SmoothL1Loss(), device="cuda")
agent = OffPolicyAgent(env=env, model= )
