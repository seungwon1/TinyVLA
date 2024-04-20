import time

import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO


print("panda-gym version:", panda_gym.__version__)


env_name = "PandaPickAndPlace-v3"

env = gym.make(env_name, render_mode="human")
observation, info = env.reset()

# load a model
model = PPO.load("data/ppo_panda_pick_and_place", env)

for _ in range(1000):
    action, _ = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
    time.sleep(0.1)

env.close()
