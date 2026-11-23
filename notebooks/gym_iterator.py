#!/usr/bin/env python
# coding: utf-8

# In[3]:


import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from stable_baselines3.common.evaluation import evaluate_policy

import wandb
import random
from stable_baselines3.common.monitor import Monitor

if 0:
    es = gym.envs.registration.registry.env_specs
    ek=list(es.keys())
    for ee in ek:
        print(ek)

env_labels = ['LunarLander-v2']
# discrete
# 'Roulette-v0', 'CartPole-v0', 'FrozenLake-v0','Taxi-v3', 'NChain-v0','FrozenLake8x8-v0',


# continu
#  'MountainCar-v0' , 'Acrobot-v1', 'Pendulum-v0',

# fails
# 'Blackjack-v0',

# Create log dir
log_dir = "gym_log/"
os.makedirs(log_dir, exist_ok=True)


for env_label in env_labels:
    hash = random.getrandbits(128)
    wandb.init(sync_tensorboard=True, project="gym_iterator",name=env_label+str("_%032x" % hash))

    env = gym.make(env_label)
    env = Monitor(env, log_dir)

    batch_agent = DQN('MlpPolicy',env, buffer_size=1000000, gamma=0.99, target_update_interval=1000, learning_starts=10000, verbose=1,
                      learning_rate=0.001, batch_size=32, exploration_final_eps = 0.01,  tensorboard_log="/tmp/teletubby")

    batch_agent.learn(total_timesteps=1000000, log_interval=4, tb_log_name="second_run",)

    print(env_label)

if 0:
    obs = env.reset()
    action, _states = batch_agent.predict(obs, deterministic=True)
    last_obs = obs
    obs, reward, done, info = env.step(action)
    batch_agent.replay_buffer.add(last_obs, obs, action, reward, done)


