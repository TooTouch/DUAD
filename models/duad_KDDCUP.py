import torch.nn as nn
import numpy as np
from .simple_ae import AutoEncoder

class DUAD_KDDCUP(nn.Module):
    def __init__(
        self, in_features : int = 119, latent_dim : int = 10, flatten_features : int = 4096):

        super(DUAD_KDDCUP, self).__init__()


        self.cosim = nn.CosineSimilarity()
        
        self.ae = AutoEncoder(in_features = in_features, latent_dim = latent_dim, flatten_features = flatten_features)

    def forward(self, x):
        z_c, x_size = self.ae.encode(x)
        x_out = self.ae.decode(z_c, x_size)
        z_r = self.cosim(
            x.flatten(start_dim=1),
            x_out.flatten(start_dim=1)
        )

        return z_c, z_r