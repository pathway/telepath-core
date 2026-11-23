#!/usr/bin/env python
# coding: utf-8

# In[3]:


import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import wandb
wandb.init(sync_tensorboard=True,project="telepath3")


env_label = 'CartPole-v0'
#env = gym.make(env_label)
env = DummyVecEnv([lambda: gym.make( env_label )])



agent = DQN('MlpPolicy', env, gamma=0.95, target_update_interval=100, learning_starts=100, verbose=1, learning_rate=0.01, batch_size=32, tensorboard_log="/tmp/teletubby")


agent.learn(total_timesteps=200000, log_interval=4, tb_log_name="second_run",)

obs = env.reset()    

action, _states = agent.predict(obs, deterministic=True)
last_obs = obs
obs, reward, done, info = env.step(action)

agent.replay_buffer.add(last_obs, obs, action, reward, done)


