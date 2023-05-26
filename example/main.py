import gymnasium as gym
import os

os.chdir("..")

from rllib.agent import DQNAgent

# from wrappers import env_wrapper
# env = env_wrapper(env)
# |%%--%%| <SdxHUdFXGS|ioCUm2CuEK>


# env = gym.make("ALE/Asteroids-v5", render_mode="human", obs_type="ram")
env = gym.make("CartPole-v1", render_mode="rgb_array")
agent = DQNAgent(
    env=env,
    lr=0.0001,
    eps=0.9,
    batch_size=128,
    gamma=0.99,
    buffer_size=10000,
    device="cuda",
)
agent.train(episodes=600)
