from coach.trainer.dqn import DQN
import gymnasium as gym


env = gym.make("CartPole-v1", render_mode="rgb_array")
trainer = DQN(env, 0.0001, 0.05, 64, 10000, 0.98, False, 0.05, "cpu")
trainer.train(600)
