from telepath_worker import spec
from telepath_gym import simplegym, batch_gym
import random

def test_specgym_basic():
    envsp = spec.EnvSpec()
    fn='specs2/spec_env_1.yaml'
    se = simplegym.SimpleGym_DiscreteAction_BoxObs()
    se.set_spec( fn )
    print(se)
    print(se.action_space)
    print(se.observation_space)

def test_specgym_extended():
    envsp = spec.EnvSpec()
    fn='specs2/spec_env_1.yaml'
    se = simplegym.SimpleGym_DiscreteAction_BoxObs()
    se.set_spec( fn )
    for i in range(0,5):
        action=0
        res = se.step( action )
        print(res,action)
    print(se)
    print(se.action_space)
    print(se.observation_space)


def test_batchgym_basic():
    g = batch_gym.BatchGym_DiscreteAction_BoxObs()
    for i in range(0,5):
        action=random.choice([0,2])
        res = g.step( action )
        print(res,action)
    print(g)
    print(g.action_space)
    print(g.observation_space)



if __name__=="__main__":
    test_batchgym_basic()
