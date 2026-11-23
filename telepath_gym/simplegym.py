from gym.envs.registration import register
import gym
import numpy as np
from gym import spaces
import random
from telepath_worker import spec

'''
Learn simulator from data.

Multiple kinds:
- Deterministic
- Stochastic
'''

class SpecEnv:

    specs = None

    def set_spec(self, spec_loc ):
        self.specs = spec.EnvSpec()
        self.specs.load(spec_loc)
        #print( self.specs.spec)
        #print(self.specs)





class SimpleGym_DiscreteAction_BoxObs(gym.Env, SpecEnv):
    # placeholder purely to describe action and action spaces
    # could be replaced by those two object, but this is convenient.

    def __init__(self,  ):
        pass

    def init(self,n_action, n_obs):
        self.action_space = spaces.Discrete(n_action)
        self.observation_space = spaces.Box(-10000,+10000,[n_obs],)

    def set_spec(self,spec_loc):
        super().set_spec(spec_loc)
        self.init( self.specs.spec['n_actions'], self.specs.spec['n_obs']  )

    def step(self, action):
        assert self.action_space.contains(action)

        # not functional; just dummy values
        reward = 0
        done = True
        reward = 1.0
        obs = 0.0
        return obs, reward, done, {}

    def reset(self):
        return 0

    def render(self, mode='human', close=False):
        pass






class SimpleGym_Sequential_Short(gym.Env ):
    '''
    Just provide simple counting observations, and 1 of n arms provides reward.
    Designed for debugging observation-shift and normalization operations.
    '''

    def __init__(self, max_steps=6, n_action=10, n_obs=3):
        self.max_steps = max_steps
        self.n_action = n_action
        self.action_space = spaces.Discrete(n_action)
        self.observation_space = spaces.Box(-10000,+10000,[n_obs],)
        self.steps = 0

    def _get_obs(self,):
        obs = []
        for i in range(0, self.observation_space.shape[0] ):
            obs.append( self.steps + i*0.1)
        return obs

    def step(self, action):
        assert self.action_space.contains(action)
        # not functional; just dummy values

        self.steps += 1

        done = False
        if action==9:
            reward = 1.0
        else:
            reward = 0.0

        obs = self._get_obs()
        #print("obs",obs,"steps",self.steps)
        #exit(1)

        if self.steps >= self.max_steps:
            done = True
            self.steps = 0

        return obs, reward, done, {}

    def reset(self):
        obs = self._get_obs()
        self.steps = 0
        return obs

    def render(self, mode='human', close=False):
        pass
