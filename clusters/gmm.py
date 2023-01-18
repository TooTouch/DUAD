from sklearn.mixture import GaussianMixture
from .base import BaseCluster
import torch

class GMM(BaseCluster):
    def __init__(
        self, p0: float = 0.25, p: float = 0.2, r: int = 5, reeval_limit: int = 10,
        num_cluster: int = 20, max_iter: int = 400):
        super(GMM, self).__init__(p0=p0, p=p, r=r, reeval_limit=reeval_limit)
        self.num_cluster = num_cluster
        self.max_iter = max_iter

        self.gmm = GaussianMixture(n_components=self.num_cluster, max_iter=self.max_iter, reg_covar=1e-5)

    def clustering(self, cluster_features: torch.Tensor):
        self.gmm.fit(cluster_features)
        cluster_pred = self.gmm.predict(cluster_features)
        cluster_centroid = torch.from_numpy(self.gmm.means_)

        return self.num_cluster, cluster_pred, cluster_centroid