import torch
import numpy as np

class BaseCluster:
    def __init__(self, p0: float = 0.25, p: float = 0.2, r: int = 5, reeval_limit: int = 10):
        self.p0 = p0
        self.p = p
        self.r = r 
        self.reeval_limit = reeval_limit

    def sampling(
        self, threshold: float, cluster_features: torch.Tensor, num_cluster: int, 
        cluster_pred: list, cluster_centroid: torch.Tensor):
        
        # calculate variance by cluster
        clusters_vars = []

        for i in range(num_cluster):
            vars_i = torch.sum((cluster_features[cluster_pred == i] - cluster_centroid[i].unsqueeze(dim=0)) ** 2)
            vars_i /= (cluster_pred == i).sum()
            clusters_vars.append(vars_i)
        
        clusters_vars = torch.tensor(clusters_vars)

        # select cluster using threshold
        q = torch.quantile(clusters_vars, 1-threshold)
        selected_clusters = (clusters_vars <= q).nonzero().squeeze()

        selection_mask = [pred in list(selected_clusters.cpu().numpy()) for pred in cluster_pred]
        selected_indices = torch.from_numpy(np.array(selection_mask)).nonzero().squeeze()

        return selected_indices