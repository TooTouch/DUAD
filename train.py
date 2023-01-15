import time
import json
import os 
import wandb
import numpy as np
import logging
from collections import OrderedDict

import torch
from sklearn.metrics import roc_auc_score

_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, dataloader, optimizer, log_interval: int, device: str) -> dict:   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    end = time.time()
    
    model.train()
    optimizer.zero_grad()

    cluster_features = torch.Tensor([])
    for idx, (inputs, targets) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        
        inputs, targets = inputs.to(device), targets.to(device)

        # predict
        z_c, z_r = model(inputs)
        z = torch.cat([z_c.flatten(start_dim=1), z_r.unsqueeze(1)],dim=1).detach().cpu()
        cluster_features = torch.cat([cluster_features, z])

        loss = 1-z_r.mean()
        loss.backward()

        # loss update
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss.item())
        
        batch_time_m.update(time.time() - end)
    
        if idx % log_interval == 0 and idx != 0: 
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        idx+1, len(dataloader), 
                        loss       = losses_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate       = inputs.size(0) / batch_time_m.val,
                        rate_avg   = inputs.size(0) / batch_time_m.avg,
                        data_time  = data_time_m))
   
        end = time.time()
    
    return OrderedDict([('loss',losses_m.avg)]), cluster_features
        
def test(model, dataloader, log_interval: int, device: str) -> dict:
    # reset
    total_targets = []
    total_score = []
    
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            
            # predict
            _, z_r = model(inputs)

            # loss 
            loss = 1-z_r.mean().item()
            
            # total loss
            total_loss += loss
            
            # update metrics
            total_score.extend(z_r.cpu().tolist())
            total_targets.extend(targets.tolist())

            if idx % log_interval == 0 and idx != 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f' % 
                            (idx+1, len(dataloader), total_loss/(idx+1)))

    # metrics    
    metrics = {
        'AUROC(image)':roc_auc_score(y_true=total_targets, y_score=total_score),
        'loss':total_loss/len(dataloader)
    }

    _logger.info('TEST: AUROC(image): %.3f%% | Loss: %.3f' % (metrics['AUROC(image)'], metrics['loss']))

    return metrics


def apply_clustering(cluster, cluster_init: int, cluster_features: torch.Tensor):
    
    num_cluster, cluster_pred, cluster_centroid = cluster.clustering(cluster_features)
    selected_indices = cluster.sampling(
        threshold        = cluster.p0 if cluster_init else cluster.p,
        cluster_features = cluster_features,
        num_cluster      = num_cluster,
        cluster_pred     = cluster_pred,
        cluster_centroid = cluster_centroid
    )
    
    return selected_indices

def fit(
    model, cluster, trainloader, testloader, optimizer, scheduler, 
    epochs: int, savedir: str, log_interval: int, device: str, use_wandb: bool
) -> None:

    best_auroc = 0
    step = 0
    cluster_init = True

    for epoch in range(epochs):
        # step scheduler
        if scheduler:
            scheduler.step()

        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        train_metrics, cluster_features = train(model, trainloader, optimizer, log_interval, device)
        eval_metrics = test(model, testloader, log_interval, device)

        # clustering and sub-sampling
        if (epoch % cluster.r) == 0:
            selected_indices = apply_clustering(
                cluster          = cluster,
                cluster_init     = cluster_init,
                cluster_features = cluster_features
            )       

            trainloader.dataset.update(select_indice=selected_indices)

            cluster_init = False

        # wandb
        if use_wandb:
            metrics = OrderedDict(
                lr          = optimizer.param_groups[0]['lr'],
                nb_trainset = len(trainloader.dataset.data)
            )
            metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
            metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
            wandb.log(metrics, step=step)

        step += 1

        # checkpoint
        if best_auroc < eval_metrics['AUROC(image)']:
            # save results
            state = {'best_epoch':epoch, 'best_auroc':eval_metrics['AUROC(image)']}
            json.dump(state, open(os.path.join(savedir, f'best_results.json'),'w'), indent=4)

            # save model
            torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
            
            _logger.info('Best AUROC {0:.3%} to {1:.3%}'.format(best_auroc, eval_metrics['AUROC(image)']))

            best_auroc = eval_metrics['AUROC(image)']

    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_auroc'], state['best_epoch']))