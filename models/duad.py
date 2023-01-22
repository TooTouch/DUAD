import torch.nn as nn
from .ae import AutoEncoder

class DUAD(nn.Module):
    def __init__(
        self, in_channels: int = 3, flatten_features: int = 4096, latent_dim: int = 10):

        super(DUAD, self).__init__()

        self.cosim = nn.CosineSimilarity()
        
        self.ae = AutoEncoder(in_channels=in_channels, flatten_features=flatten_features, latent_dim=latent_dim)

    def forward(self, x):
        z_c, x_size = self.ae.encode(x)
        x_out = self.ae.decode(z_c, x_size)
        z_r = self.cosim(
            x.flatten(start_dim=1),
            x_out.flatten(start_dim=1)
        )

        return z_c, z_r
