#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
from stable_baselines3 import DQN, BatchDQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


env_label = 'CartPole-v0'
env = gym.make(env_label)


# In[4]:


batch_agent = BatchDQN('MlpPolicy', env, verbose=0, learning_rate=0.0001, batch_size=2048)


# In[ ]:


obs = env.reset()    


# In[ ]:


action, _states = batch_agent.predict(obs, deterministic=True)
last_obs = obs
obs, reward, done, info = env.step(action)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# 1M inserts in 6s\n\nfor i in range(0,10000):\n    batch_agent.replay_buffer.add(last_obs, obs, action, reward, done)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# 100k x 100 samples : 1.3s\n# 100k x 1000 samples : 4.6s\n\nfor i in range(0,100000):\n    batch_agent.replay_buffer._get_samples_raw(1000)')


# In[ ]:


batch_agent.replay_buffer._get_samples_raw(1)


# In[ ]:


batch_agent.replay_buffer.sample(1)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# 100k x 100 samples :   5.8s\n# 100k x 1000 samples :  9.5s\n\nfor i in range(0,100000):\n    batch_agent.replay_buffer.sample(100)')


# In[1]:


import vaex
import pandas as pd


# In[2]:


import vaex.ml

vdf = vaex.ml.datasets.load_iris()
vdf  # Since this is the last statement in a cell, it will print the DataFrame in a nice HTML format.
vdf.describe()


# In[3]:


vdf.to_pandas_df()


# In[4]:


vdf.plot1d('sepal_length',shape=10)


# In[5]:


vdf.plot('sepal_length', 'petal_length',shape=10)


# In[6]:


import vaex.ml
import vaex.ml.lightgbm
df = vaex.ml.datasets.load_iris()

features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
df_train, df_test = df.ml.train_test_split()
params = {
    'boosting': 'gbdt',
    'max_depth': 5,
    'learning_rate': 0.1,
    'application': 'multiclass',
    'num_class': 3,
    'subsample': 0.80,
    'colsample_bytree': 0.80}
booster = vaex.ml.lightgbm.LightGBMModel(features=features, target='class_', num_boost_round=100, params=params)
booster.fit(df_train)


# In[7]:


df_train = booster.transform(df_train)
df_train.head(20)


# In[8]:


import sys
get_ipython().system('jupyter nbextension enable --sys-prefix --py widgetsnbextension')
get_ipython().system('jupyter nbextension enable --sys-prefix --py bqplot')
get_ipython().system('jupyter nbextension enable --sys-prefix --py ipyvolume')
get_ipython().system('jupyter nbextension enable --sys-prefix --py ipympl')
get_ipython().system('jupyter nbextension enable --sys-prefix --py ipyleaflet')


# In[9]:


#df = vaex.example()

df.plot_widget(df.sepal_width, df.petal_length,  backend='bqplot')


# In[ ]:


df.plot_widget(df.sepal_width, df.petal_length, df.petal_width, backend='ipyvolume')


# In[ ]:


df.plot_widget(df.sepal_width, df.petal_length, df.petal_width, backend='ipympl')


# In[ ]:


df.plot_widget(df.sepal_width, df.petal_length, df.petal_width, backend='widgetsnbextension')


# In[ ]:




