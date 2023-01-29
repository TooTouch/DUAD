import gzip
import pickle

import numpy as np
import torch

from torch import nn
from typing import Tuple, List

class AutoEncoder(nn.Module):
    def __init__(self, in_features : int = 119, latent_dim : int = 10, flatten_features : int = 4096):
        super(AutoEncoder, self).__init__()

        # self.latent_dim = dec_layers[0][0]
        # self.in_features = enc_layers[-1][1]
        
        self.in_features = in_features
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_features, 60),
            nn.Tanh(),
            nn.Linear(60, 30),
            nn.Tanh(),
            nn.Linear(30, 10),
            nn.Tanh(),
            nn.Linear(10, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.Tanh(),
            nn.Linear(10, 30),
            nn.Tanh(),
            nn.Linear(30, 60),
            nn.Tanh(),
            nn.Linear(60, in_features)
        )

#        self.enc_linear = nn.Linear(flatten_features, latent_dim)
#        self.dec_linear = nn.Linear(latent_dim, flatten_features)

    def encode(self, x):
        x_out = self.encoder(x) 
        x_size = x_out.size()

#        z = self.enc_linear(x_out.flatten(start_dim=1))
        z = x_out.flatten(start_dim = 1)

        return z, x_size

    def decode(self, z, x_size):
#        z = self.dec_linear(z)
        x = z.view(x_size)

        out = self.decoder(x)

        return out