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
'''
class SpecEnv:

    specs = None

    def set_spec(self, spec_loc ):
        self.specs = spec.EnvSpec()
        self.specs.load(spec_loc)
        print( self.specs.spec)
        #print(self.specs)
'''




class BatchGym_DiscreteAction_BoxObs(gym.Env,):
    '''
    Purpose:
    - testbed for Q value explosion
    - usage: batch data, filter out specific action
        - inspect Q values for missing action
        - look for Q value explosion
        - compare algos

    Simple env:
    - ignores observation
    - constant reward per action like bandit
    - BUT multiplestep MDP, allowing Q values to grow
    '''

    def __init__(self, max_steps=5, action_reward_vector=[ 3, 0, 4, 0 ] ):
        self.action_reward_vector = action_reward_vector
        self.n_action = len(action_reward_vector)
        self.max_steps = max_steps

        self.init(self.n_action, n_obs=1)
        self.reset()

    def init(self,n_action, n_obs):
        self.action_space = spaces.Discrete(n_action)
        self.observation_space = spaces.Box(-10000,+10000,[n_obs],)

    def step(self, action):
        assert self.action_space.contains(action)

        self.steps+=1

        reward = self.action_reward_vector[action] ** self.steps
        obs = [  float(self.steps) ]

        if self.steps >= self.max_steps:
            done = True
            self.reset()
        else:
            done = False

        return obs, reward, done, {}

    def reset(self):
        self.steps=0
        return 0

    def render(self, mode='human', close=False):
        pass


class BatchGym_DiscreteAction_BoxObs_Modulo(gym.Env,):
    '''
    Purpose:
    - testbed for Q value explosion
    - usage: batch data, filter out specific action
        - inspect Q values for missing action
        - look for Q value explosion
        - compare algos

    Simple env:
    - ignores observation
    - constant reward per action like bandit
    - BUT multiplestep MDP, allowing Q values to grow
    '''

    def __init__(self, max_steps=5, action_reward_vector=[ 3, 2, 7, -1 ] ):
        self.action_reward_vector = action_reward_vector
        self.n_action = len(action_reward_vector)
        self.max_steps = max_steps

        self.init(self.n_action, n_obs=1)
        self.reset()

    def init(self,n_action, n_obs):
        self.action_space = spaces.Discrete(n_action)
        self.observation_space = spaces.Box(-10000,+10000,[n_obs],)

    def step(self, action):
        assert self.action_space.contains(action)

        self.steps+=1

        obs = [  float(self.steps) ]

        reward = self.action_reward_vector[ (action + self.steps) % self.n_action   ]

        #print(" action obs mod action_val reward", action, obs, mod, self.action_reward_vector[action], reward)

        if self.steps >= self.max_steps:
            done = 1
            self.reset()
        else:
            done = 0

        return obs, reward, done, {}

    def reset(self):
        self.steps=0
        return [ float(self.steps) ]

    def render(self, mode='human', close=False):
        pass



if __name__=="__main__":
    g = BatchGym_DiscreteAction_BoxObs_Modulo()
    for a in range(0,20):
        act = random.randint(0,3)
        print(act,g.step(act))
    print(g.step(2))
    print(g.step(0))
    print(g.step(0))
    print(g.step(2))
    print(g.step(2))




