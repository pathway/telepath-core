
'''
from telepath.deepgym import DeepGym_DiscreteAction_DiscreteObs

from gym.envs.registration import register

register(
    id='DeepGym_DiscreteAction_BoxObs-v0',
    entry_point='telepath.deepgym:DeepGym_DiscreteAction_BoxObs',
)
'''
from telepath_worker import *
import telepath_gym

import gym
from telepath_gym.batch_gym import BatchGym_DiscreteAction_BoxObs

gym.envs.register(
     id='BatchGym_DiscreteAction_BoxObs-v0',
     entry_point='telepath_gym.batch_gym:BatchGym_DiscreteAction_BoxObs',
     #max_episode_steps=150,
     #kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : np.inf},
)

gym.envs.register(
     id='BatchGym_DiscreteAction_BoxObs_Modulo-v0',
     entry_point='telepath_gym.batch_gym:BatchGym_DiscreteAction_BoxObs_Modulo',
     #max_episode_steps=150,
     #kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : np.inf},
)


gym.envs.register(
     id='SimpleGym_Sequential_Short-v0',
     entry_point='telepath_gym.simplegym:SimpleGym_Sequential_Short',
     #max_episode_steps=150,
     #kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : np.inf},
)

