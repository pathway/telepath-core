import os
import yaml
import argparse
from datetime import datetime
import gym
from fqf.fqf_iqn_qrdqn.env import make_pytorch_env
from fqf.fqf_iqn_qrdqn.agent import IQNAgent
import fqf.fqf_iqn_qrdqn
import gym

iscuda=True
args_seed=42
args_config="config/iqn_flatgym.yaml"
with open(args_config) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

env_id="CartPole-v0"
env = gym.make(env_id)
test_env = env

time = datetime.now().strftime("%Y%m%d-%H%M")
log_dir = os.path.join( 'logs', env_id, f'IQN-{args_seed}-{time}')

# Create the agent and run.
agent = IQNAgent(
env=env, test_env=test_env, log_dir=log_dir, seed=args_seed,
cuda=iscuda, **config)
agent.run()

agent.save_models('savedir')
