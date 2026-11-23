import pandas as pd

import time
import torchvision
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm

from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch


from telepath_worker import ext_replay
from telepath_worker import spec

import deepgym_nets


def get_fwd_result(model, action, obs):
    '''
    :action int:
    :obs: list of floats
    '''
    action_tens = torch.tensor([[action]])
    obs_tens = torch.tensor([ obs ])
    nx_obs, reward, done =  model.forward(obs_tens,action_tens)

    if done > 0.5:
        done = True
    else:
        done = False
    return nx_obs, round(reward.item(),3), done


def train_model( model, r, device, num_epochs = 10 ):
    since = time.time()
    criterion = nn.MSELoss()
    criterion_bin = nn.BCELoss()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('lr',model.optimizer.param_groups[0]['lr'])

        print('-' * 10)
        model.scheduler.step(1.)
        model.train()
        running_loss = 0.0
        counter = 0
        batch_size=128
        data_rows=5000
        for i in range(0, int(data_rows / batch_size) ):
            s = r.sample(batch_size=batch_size)

            obs = s.observations.to(device, dtype=torch.float)
            lab_nx_obs = s.next_observations.to(device, dtype=torch.float)
            lab_rew = s.rewards.to(device, dtype=torch.float)
            lab_action = s.actions.to(device, dtype=torch.float)
            lab_done = s.dones.to(device, dtype=torch.float)

            model.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                nx_obs, rew, done = model(obs, lab_action)

                nx_obs=nx_obs.to(device, dtype=torch.float)
                rew=rew.to(device, dtype=torch.float)
                done=done.to(device, dtype=torch.float)

                # print( "nx_obs_loss",criterion(nx_obs, lab_nx_obs))
                # print( "rew_loss",criterion(lab_rew,rew))
                # print( "done_loss",criterion(lab_done,done))

                loss_lin = criterion(nx_obs, lab_nx_obs) + criterion(lab_rew,rew)
                loss_bin = criterion_bin(done , lab_done)
                if i%10==0:
                    print( "loss_lin, loss_bin" ,loss_lin, loss_bin)
                loss = loss_lin + loss_bin
                loss.backward()
                model.optimizer.step()

            running_loss += loss.item() * obs.size(0)
            counter += 1
            #tk0.set_postfix(loss=(running_loss / (counter * data_loader.batch_size)))
        epoch_loss = running_loss # / len(data_loader)
        print('Training Loss: {:.4f}'.format(epoch_loss))

        print_sample_fwd(model=model)

    return model

def print_sample_fwd(model):
    action = 9
    obs = [3.0, 3.1, 3.2]
    nx_obs, reward, done = get_fwd_result(model=model, action=action, obs=obs)
    print("nx_obs, reward, done",nx_obs, reward, done )


def save_model(model):
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.save(model.state_dict(), "model.bin")



def load_replay(replay_spec_filepath, replay_filepath ):
    repsp = spec.ReplaySpec()
    repsp.load(replay_spec_filepath)
    print(repsp.spec)

    r = ext_replay.VxBuffer(replay_filepath)
    r.sample_strategy = 'seq'
    r.fieldmap = repsp.spec['fieldmap']
    r.read_from_location()

    print(r)
    print(r.df)
    print(r.sample(2))
    # exit(1)
    return r, repsp


def build_model( n_actions, n_obs, device_string ):
    device = torch.device(device_string)
    model = deepgym_nets.DeepGym_ModelNet(n_actions=n_actions, n_obs=n_obs, device_string=device_string)
    model = model.to(device)
    print(model)
    return model


# -- main
def _test_deepgym():
    device_string="cuda:0"

    r, repsp = load_replay(  replay_spec_filepath='../telepath_worker/specs/simplegym_seq_short/spec_replay.yaml' , replay_filepath='/path/to/telepath-core/projects/simplegym_seq//replay.csv' )

    n_obs = len(repsp.spec['fieldmap']['obs_cols'])
    n_actions = repsp.spec['n_actions']

    model = build_model( n_actions = n_actions, n_obs=n_obs, device_string=device_string )
    model = train_model( model, r, device=device_string, num_epochs=1000)
    print_sample_fwd(model=model)


_test_deepgym()
