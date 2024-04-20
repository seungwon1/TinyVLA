import gymnasium as gym
import panda_gym  # noqa: F401
from stable_baselines3 import A2C

env = gym.make("PandaReachDense-v3")
model = A2C(policy="MultiInputPolicy", env=env, verbose=1)

model.learn(total_timesteps=1_000_000, progress_bar=True)
model.save("data/a2c_panda_reach")
