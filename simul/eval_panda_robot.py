# %%
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import A2C
import gymnasium as gym
from moviepy.editor import ImageSequenceClip
from IPython.display import Image
import panda_gym

# %%

env_id = "PandaReachDense-v3"
model = A2C.load("a2c_panda_reach")
env = gym.make(env_id, render_mode="rgb_array")
obs, info = env.reset()
images = [env.render()]

for _ in range(1000):
    action, _state = model.predict(obs)
    osb, reward, done, truncated, info = env.step(action)
    images.append(env.render())

    if done or truncated:
        observation, info = env.reset()
        images.append(env.render())


env.close()
# %%
fps = 40
clip = ImageSequenceClip(images, fps=fps)
clip.write_gif("evaluated.gif", fps=fps)
Image(filename="evaluated.gif")
