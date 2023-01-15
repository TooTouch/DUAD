from .duad import DUAD

def create_model(in_channels: int = 3, flatten_features: int = 4096, latent_dim: int = 10):
    return DUAD(in_channels=in_channels, flatten_features=flatten_features, latent_dim=latent_dim)