#!/usr/bin/env python
# coding: utf-8

# In[3]:


import gym
from stable_baselines3 import DQN, BatchDQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


# In[5]:


env_label = 'CartPole-v0'
env = gym.make(env_label)


# In[6]:


batch_agent = BatchDQN('MlpPolicy', env, verbose=0, learning_rate=0.0001, batch_size=2048)


# In[7]:


obs = env.reset()    


# In[9]:


action, _states = batch_agent.predict(obs, deterministic=True)
last_obs = obs
obs, reward, done, info = env.step(action)


# In[11]:


batch_agent.replay_buffer.add(last_obs, obs, action, reward, done)


# In[ ]:





# In[ ]:





# In[ ]:




