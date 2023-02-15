import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import pickle as pl

class DUADDataset:
    def train_preprocessing(self):
        np.random.seed(self.random_state)
        
   ```` # select normal
        normal_indices = np.where(np.array(self.targets)==self.normal_class)[0]
        data = self.data[normal_indices]
        targets = np.zeros(normal_indices.shape)


        # sampling abnormal
        for t_idx in np.unique(targets):
            if t_idx != self.normal_class:
                abnormal_indices = np.where(np.array(self.targets)==t_idx)[0]
                abnormal_indices = np.random.choice(abnormal_indices, size=self.abnormal_sample_size, replace=False)
        
                data = np.concatenate([data, self.data[abnormal_indices]])
                targets = np.concatenate([targets, np.ones(abnormal_indices.shape)])
        
        setattr(self,'data', data)
        setattr(self,'targets', targets)

    def test_preprocessing(self):
        targets = np.zeros(len(self.targets))
        targets[np.where(np.array(self.targets)!=self.normal_class)[0]] = 1

        setattr(self, 'targets', targets)

    def update(self, select_indice):
        self.data = self.data[select_indice]
        self.targets = self.targets[select_indice]

class KDDCupData(DUADDataset):
    def __init__(self, data_dir, mode,  normal_class: int = 0, abnormal_sample_size: int = 12847, random_state: int = 42):
        """Loading the data for train and test."""
        data = np.load(data_dir, allow_pickle=True)

        labels = data["kdd"][:,-1]
        features = data["kdd"][:,:-1]
        # In this case, "attack" has been treated as normal data as is mentioned in the paper
        normal_data = features[labels==0] 
        normal_labels = labels[labels==0]

        # 정상데이터의 0.5만큼을 샘플링하여 test 데이터로 이용
        n_train = int(normal_data.shape[0]*0.5)
        ixs = np.arange(normal_data.shape[0])
        np.random.shuffle(ixs)
        normal_data_test = normal_data[ixs[n_train:]]
        normal_labels_test = normal_labels[ixs[n_train:]]

        DUADDataset.__init__(self)

        self.normal_class = normal_class
        self.abnormal_sample_size = abnormal_sample_size
        self.random_state = random_state

        if mode == 'train':
            self.data = normal_data[ixs[:n_train]]
            self.targets = normal_labels[ixs[:n_train]]
            self.train_preprocessing()

        elif mode == 'test':
            anomalous_data = features[labels==1]
            anomalous_labels = labels[labels==1]
            self.data = np.concatenate((anomalous_data, normal_data_test), axis=0)
            self.targets = np.concatenate((anomalous_labels, normal_labels_test), axis=0)
            self.test_preprocessing()

    def __len__(self):
        """ Number of images in the object dataset."""
        return self.data.shape[0]

    def __getitem__(self, index):
        """ Return a sample from the dataset."""
        data_i, target_i = self.data[index], self.targets[index]
        return np.float32(data_i), np.float32(target_i)
    
def get_KDDCup99(args, data_dir='./kdd_cup.npz'):
    """ Returning train and test dataloaders."""
    train = KDDCupData(data_dir, 'train')
    dataloader_train = DataLoader(train, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    
    test = KDDCupData(data_dir, 'test')
    dataloader_test = DataLoader(test, batch_size=args.batch_size, 
                              shuffle=False, num_workers=0)
    return dataloader_train, dataloader_test