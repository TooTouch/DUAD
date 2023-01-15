from .duad import DUAD

def create_model(in_channels: int = 3):
    return DUAD(in_channels=in_channels)