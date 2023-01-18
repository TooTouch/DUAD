from torchvision import datasets
import numpy as np
from torchvision import transforms

class DUADDataset:
    def train_preprocessing(self):
        np.random.seed(self.random_state)
        
        # select normal
        normal_indices = np.where(np.array(self.targets)==self.normal_class)[0]
        data = self.data[normal_indices]
        targets = np.zeros(normal_indices.shape)
        
        # sampling abnormal
        for t_idx in self.class_to_idx.values():
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


class CIFAR10Dataset(DUADDataset, datasets.CIFAR10):
    def __init__(
        self, root, train: bool = True, download: bool = True,
        normal_class: int = 0, abnormal_sample_size: int = 50, random_state: int = 42
    ):
        datasets.CIFAR10.__init__(
            self, root=root, train=train, download=download
        )    
        DUADDataset.__init__(self)

        self.normal_class = normal_class
        self.abnormal_sample_size = abnormal_sample_size
        self.random_state = random_state

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        if train:
            self.train_preprocessing()
        else:
            self.test_preprocessing()
        
    def __getitem__(self, idx):
        
        data_i, target_i = self.data[idx], self.targets[idx]

        return self.transform(data_i), target_i
        
        
class MNISTDataset(DUADDataset, datasets.MNIST):
    def __init__(
        self, root, train: bool = True, download: bool = True,
        normal_class: int = 4, abnormal_sample_size: int = 30, random_state: int = 42
    ):
        datasets.MNIST.__init__(
            self, root=root, train=train, download=download
        )    
        DUADDataset.__init__(self)

        self.normal_class = normal_class
        self.abnormal_sample_size = abnormal_sample_size
        self.random_state = random_state
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        
        if train:
            self.train_preprocessing()
        else:
            self.test_preprocessing()
        
    def __getitem__(self, idx):
        
        data_i, target_i = self.data[idx], self.targets[idx]

        return self.transform(data_i), target_i
        