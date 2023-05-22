import gymnasium as gym
import os

os.chdir("..")
print(os.getcwd())

from rllib.agent import DQNAgent

# from wrappers import env_wrapper
# env = env_wrapper(env)
# |%%--%%| <SdxHUdFXGS|ioCUm2CuEK>


# env = gym.make("ALE/Asteroids-v5", render_mode="human", obs_type="ram")
env = gym.make("CartPole-v1", render_mode="human")
env.metadata["render_fps"] = 15
agent = DQNAgent(env=env, verbose="debug")
agent.train()
