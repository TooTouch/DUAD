from sklearn.mixture import GaussianMixture
from .base import BaseCluster
import torch
import scipy 
import numpy as np 
from tqdm import tqdm 

class DistributionClustering(BaseCluster):
    def __init__(
        self, p0: float = 0.25, p: float = 0.2, r: int = 5, reeval_limit: int = 10,num_cluster=20, max_iter: int = 400):
        super(DistributionClustering, self).__init__(p0=p0, p=p, r=r, reeval_limit=reeval_limit)
        self.max_iter = max_iter
        
    def clustering(self,cluster_features: torch.Tensor):
        centers, clusters, cluster_distances = cluster(cluster_features)
        return centers, clusters, cluster_distances
        
        

def cluster(features, thres=0.05, min_clus=5, max_dist=0.02, normalize=True):
    '''
    max_dist : clustering 전체 종료 조건 
    thres : clustering 할당 종료 조건 
    '''
    
    assert len(features) > 0 and len(features.shape) == 2

    if normalize:
        feat_norms = np.linalg.norm(features, axis=1, keepdims=True)
        feat_norms[feat_norms == 0] = 1
        features /= feat_norms
    
    pair_dist = scipy.spatial.distance.pdist(features, 'sqeuclidean')
    pair_dist = scipy.spatial.distance.squareform(pair_dist)
    
    # Loop initialization
    inf = 1000.0
    pair_dist_base = pair_dist.copy()
    pair_dist = pair_dist + inf * np.identity(len(features))
    sample_clusters = np.zeros(len(features), dtype=np.int)
    cluster_distances = {}
    cur_cluster = 1
    finished = False
    
    print('\n----------------')
    print('클러스터링 시작')
    while not finished:
        print(f' 현재 클러스터 : {cur_cluster}')
        finished = True
        if (sample_clusters > 0).sum() < len(features):
            i, j = np.unravel_index(pair_dist.argmin(), pair_dist.shape)
            cur_dist = pair_dist[i, j]
            pair_dist[i, j] = pair_dist[j, i] = inf # 다음에 안걸리게 최대치로 변경 
            cur_vec = compute_vec(pair_dist_base, [i, j], cur_dist) # i,j의 고유 벡터 

            a, b = pair_dist_base[i, :].copy(), pair_dist_base[j, :].copy()
            '''
            a : i와 나머지 다른 것들 간의 거리 벡터 
            b : j와 나머지 다른 것들 간의 거리 벡터 
            '''
            
            a[j] = b[i] = 0 
            if np.abs(a - b).mean() > thres and cur_dist <= max_dist:
                '''
                a,b 는 가장 가까운 벡터 두개의 pair인데 이게 thres보다 먼 경우 
                하나의 cluster가 아니라고 판단하여 break 함 
                '''
                finished = False
                continue

            if cur_dist == 0:
                finished = False
                continue

            if cur_dist <= max_dist: 
                '''
                최소 거리가 max_dist보다 낮은 경우 해당 pair는 하나의 클러스터에 있다고 판단
                클러스터링 할당 진행 
                '''
                clus = sample_clusters.copy()
                
                clus[i] = clus[j] = cur_cluster
                '''
                우선 최소 거리 pair i,j에 cur_cluster 번호 할당 
                '''
                clus = clus_strange(pair_dist_base, clus, cur_vec, cur_dist, thres, thres, cur_cluster)

                loc_ind = np.argwhere(clus == cur_cluster).flatten()

                cur_dist = compute_dist(pair_dist_base, loc_ind)
                cur_vec = compute_vec(pair_dist_base, loc_ind, cur_dist)
                clus = clus_strange(pair_dist_base, clus, cur_vec, cur_dist, thres, thres, cur_cluster)

                if (clus == cur_cluster).sum() > min_clus:
                    
                    sample_clusters = clus
                    cluster_distances[cur_cluster] = cur_dist
                    cur_cluster += 1
                    pair_dist[sample_clusters > 0, :] = pair_dist[:, sample_clusters > 0] = inf

                finished = False

        if cur_dist > max_dist:
            finished = True

    if normalize:
        features *= feat_norms

    centers = [features[sample_clusters == c].mean(axis=0) for c in range(1, max(sample_clusters) + 1)]
    '''
    centers : features에서 해당 cluster에 속한 것의 가장 중간에 해당하는 벡터 값 -> centroid 
    
    cluster_distances : 해당 cluster 내에서 가장 짧은 pair의 거리 
    '''    
    
    return centers, sample_clusters.tolist(), cluster_distances





def compute_vec(d, indices, dist):
    d = d[indices, :].copy()
    for i in range(len(indices)):
        d[i, indices[i]] = dist
    return d.mean(axis=0)


def compute_dist(d, indices):
    d = d[indices, :][:, indices]
    s = len(d)
    return d.sum() / (s ** 2 - s)


def clus_strange(d, clus, cur_vec, cur_dist, thres_all, thres_loc, cur_clus):
    '''
    d                    : pair_distance matrix 
    clus                 : cluster number list
    cur_vec              : 해당 클러스터 최소 거리 pair의 고유 벡터 
    cur_dist             : 아직 할당되지 않은 features 간의 거리 중 가장 짧은 거리 
    thres_all, thres_loc : threshold 
    cur_clus             : 현재 할당 cluster 번호 
    '''
    change = True
    while change:
        change = False
        for i in tqdm(range(len(clus))):
            if clus[i] != cur_clus: # 할당 되지 않은 것이 있다면 진행
                continue
            for j in range(len(clus)):
                if i == j or clus[j] > 0: 
                    continue
                
                vec_j = d[j, :].copy() # j features와 나머지 간의 거리 vector 
                vec_j[j] = cur_dist
                
                dist = np.abs(cur_vec - vec_j).mean()  
                t = d[j, :][clus == cur_clus].mean() 
                '''
                dist : cur_clus의 고유 벡터 간의 거리 평균 
                t    : j의 벡터와 현재 cur_clus로 할당되어 있는 벡터들 간의 거리 평균 
                '''
                if dist < thres_all and abs(t - cur_dist) < thres_loc:
                    clus[j] = clus[i]
                    change = True
    return clus
