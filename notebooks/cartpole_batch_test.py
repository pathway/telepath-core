#!/usr/bin/env python
# coding: utf-8

# In[3]:


import gym
from stable_baselines3 import DQN, BatchDQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

#import wandb
#wandb.init(sync_tensorboard=True,project="telepath3")


env_label = 'CartPole-v0'
env = gym.make(env_label)



batch_agent = BatchDQN('MlpPolicy', env, gamma=0.9, target_update_interval=10, learning_starts=100, verbose=1, learning_rate=0.33, batch_size=32, tensorboard_log="/tmp/teletubby")


batch_agent.learn_batch(total_timesteps=200000, log_interval=4, tb_log_name="second_run",)

obs = env.reset()    



action, _states = batch_agent.predict(obs, deterministic=True)
last_obs = obs
obs, reward, done, info = env.step(action)



batch_agent.replay_buffer.add(last_obs, obs, action, reward, done)


