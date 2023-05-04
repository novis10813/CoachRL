import gymnasium as gym

from agent import DQN
from wrappers import env_wrapper

env =gym.make("ALE/Asteroids-v5", render_mode="human", obs_type="ram")
# env = env_wrapper(env)
env.metadata['render_fps'] = 15
