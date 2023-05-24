import gymnasium as gym
import os

os.chdir("..")

from rllib.agent import DQNAgent

# from wrappers import env_wrapper
# env = env_wrapper(env)
# |%%--%%| <SdxHUdFXGS|ioCUm2CuEK>


# env = gym.make("ALE/Asteroids-v5", render_mode="human", obs_type="ram")
env = gym.make("CartPole-v1", render_mode="human")
# print(env.observation_space.sample())
# print(env.observation_space)
env.metadata["render_fps"] = 15
env.reset()
env.action_space.sample()
agent = DQNAgent(env=env)
agent.train()
