import torch.nn as nn
from .ae import AutoEncoder

class DUAD(nn.Module):
    def __init__(
        self, r: int = 10, p0: int = .35, p: int = .30):

        super(DUAD, self).__init__()

        self.p0 = p0
        self.p = p
        self.r = r
        self.cosim = nn.CosineSimilarity()

        self.ae = AutoEncoder()

    def forward(self, x):
        z_c = self.ae.encoder(x)
        x_out = self.ae.decoder(z_c)
        z_r = self.cosim(
            x.flatten(start_dim=1),
            x_out.flatten(start_dim=1)
        )

        return z_c, z_r, x_out
