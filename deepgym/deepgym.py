from gym.envs.registration import register
import gym
import numpy as np
from gym import spaces
import random

'''
Learn simulator from data.

Multiple kinds:
- Deterministic
- Stochastic
'''

class DeepGym_DiscreteAction_BoxObs(gym.Env):

    def __init__(self, n_action, n_obs ):
        self.action_space = spaces.Discrete(n_action)
        self.observation_space = spaces.Box(-10000,+10000,[n_obs],)

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0
        done = True
        reward = random.randint()
        return 0, reward, done, {}

    def reset(self):
        return 0

    def learn_from_buffer(self, buffer):
        # network
        # sampler
        # train loop
        pass

    def render(self, mode='human', close=False):
        pass



def load_deepgym( path, n_action, n_obs ):
    return DeepGym_DiscreteAction_BoxObs( n_action, n_obs )

