from .duad import DUAD
from .duad_KDDCUP import DUAD_KDDCUP

def create_model(in_channels: int = 3, flatten_features: int = 4096, latent_dim: int = 10):
    return DUAD(in_channels=in_channels, flatten_features=flatten_features, latent_dim=latent_dim)

def create_kddcup_model(in_features : int = 119, latent_dim : int = 10, flatten_features : int = 4096):
    return DUAD_KDDCUP(in_features = in_features, latent_dim = latent_dim, flatten_features = flatten_features)