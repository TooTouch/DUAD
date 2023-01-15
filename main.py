import wandb
import logging
import os
import torch
import argparse
import yaml

from data import create_dataset, create_dataloader
from models import create_model
from clusters import create_cluster
from train import training
from log import setup_default_logging
from utils import torch_seed
from scheduler import CosineAnnealingWarmupRestarts


_logger = logging.getLogger('train')


def run(cfg):
    # savedir
    cfg['EXP_NAME'] = cfg['EXP_NAME'] + f"-{cfg['DATASET']['target']}"
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['EXP_NAME'])
    os.makedirs(savedir, exist_ok=True)

    # setting seed and device
    setup_default_logging(log_path=savedir)
    torch_seed(cfg['SEED'])

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))
    
    # wandb
    if cfg['TRAIN']['use_wandb']:
        wandb.init(name=cfg['EXP_NAME'], project='MemSeg', config=cfg)

    # build datasets
    trainset, testset = create_dataset(
        datadir  = cfg['DATASET']['datadir'],
        dataname = cfg['DATASET']['dataname']
    )
    
    # build dataloader
    trainloader = create_dataloader(
        dataset     = trainset,
        train       = True,
        batch_size  = cfg['TRAIN']['batch_size'],
        num_workers = cfg['TRAIN']['num_workers']
    )
    
    testloader = create_dataloader(
        dataset     = testset,
        train       = False,
        batch_size  = cfg['TRAIN']['batch_size'],
        num_workers = cfg['TRAIN']['num_workers']
    )


    # build feature extractor
    model = create_model(
        in_channels      = cfg['MODEL']['in_channels'],  
        flatten_features = cfg['MODEL']['flatten_features'],
        latent_dim       = cfg['MODEL']['latent_dim']
    ).to(device)

    # build clustsering method
    cluster = create_cluster(
        name   = cfg['CLUSTER']['name'],
        params = cfg['CLUSTER']['parameters'],
        p0     = cfg['CLUSTER']['p0'],
        p      = cfg['CLUSTER']['p'],
        r      = cfg['CLUSTER']['r'],
    )

    # set optimizer
    optimizer = torch.optim.SGD(
        params = model.parameters(), 
        lr     = cfg['OPTIMIZER']['lr'],
    )

    if cfg['SCHEDULER']['use_scheduler']:
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, 
            first_cycle_steps = cfg['TRAIN']['num_training_steps'],
            max_lr = cfg['OPTIMIZER']['lr'],
            min_lr = cfg['SCHEDULER']['min_lr'],
            warmup_steps   = int(cfg['TRAIN']['num_training_steps'] * cfg['SCHEDULER']['warmup_ratio'])
        )
    else:
        scheduler = None

    # Fitting model
    training(
        model              = model, 
        cluster            = cluster,
        num_training_steps = cfg['TRAIN']['num_training_steps'], 
        trainloader        = trainloader, 
        validloader        = testloader, 
        optimizer          = optimizer,
        scheduler          = scheduler,
        log_interval       = cfg['LOG']['log_interval'],
        eval_interval      = cfg['LOG']['eval_interval'],
        savedir            = savedir,
        device             = device,
        use_wandb          = cfg['TRAIN']['use_wandb']
    )


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DUAD')
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')    

    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    run(cfg)