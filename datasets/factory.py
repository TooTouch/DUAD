import os
from torch.utils.data import DataLoader


def create_dataset(datadir: str, dataname: str):
    trainset = __import__('datasets').__dict__[f'{dataname}Dataset'](
        root      = os.path.join(datadir,dataname), 
        train     = True, 
        download  = True
    )
    testset = __import__('datasets').__dict__[f'{dataname}Dataset'](
        root      = os.path.join(datadir,dataname), 
        train     = False, 
        download  = True
    )

    return trainset, testset


def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = False, num_workers: int = 0):

    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers
    )
