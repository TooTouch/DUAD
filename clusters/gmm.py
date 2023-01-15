from sklearn.mixture import GaussianMixture
from .base import BaseCluster
import torch

class GMM(BaseCluster):
    def __init__(
        self, p0: float = 0.25, p: float = 0.2, r: int = 5, 
        num_clusters: int = 10, max_iter: int = 400):
        super(GMM, self).__init__(p0=p0, p=p, r=r)
        self.num_clusters = num_clusters
        self.max_iter = max_iter

        self.gmm = GaussianMixture(n_components=self.num_clusters, max_iter=self.max_iter)

    def clustering(self, cluster_features: torch.Tensor):
        self.gmm.fit(cluster_features)
        cluster_pred = self.gmm.predict(cluster_features)
        cluster_centroid = torch.from_numpy(self.gmm.means_)

        return self.num_cluster, cluster_pred, cluster_centroid