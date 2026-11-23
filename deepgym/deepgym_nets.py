import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepGym_ModelNet(nn.Module):

    def __init__(self, n_actions, n_obs, model_latent_size=1024, device_string="cuda:0" ):
        super(DeepGym_ModelNet, self).__init__()
        self.n_obs = n_obs
        self.n_actions = n_actions

        self.fc1 = nn.Linear(n_obs + n_actions, model_latent_size)
        self.fc2 = nn.Linear(model_latent_size, model_latent_size)

        self.out_obs = nn.Linear(model_latent_size, n_obs)
        self.out_r = nn.Linear(model_latent_size, 1)
        self.out_d_lin = nn.Linear(model_latent_size, 1)
        self.out_d = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=100)

        self.device = torch.device(device_string)

    def forward(self, obs, action):
        batch_size = action.shape[0]
        one_hot = torch.zeros(batch_size, self.n_actions, dtype=torch.long)
        action = action.type(torch.cuda.LongTensor)
        target=one_hot.scatter_(1, action.to('cpu'), 1)        # TODO: fix this one-hot encoding
        #print(target.detach().cpu())
        #print(one_hot)
        inp = torch.cat( [obs.T.to(self.device), target.T.to(self.device) ] ).T     # TODO: yuck
        x = F.relu(self.fc1(  inp ))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))

        nx_obs = self.out_obs(x)
        reward = self.out_r(x)
        done = self.out_d( self.out_d_lin(x) )
        return nx_obs, reward, done
