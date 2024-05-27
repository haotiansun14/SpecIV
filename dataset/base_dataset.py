'''
This file is used to generate experimental data based on the dSprites dataset.
Original dataset: L. Matthey, I. Higgins, D. Hassabis, and A. Lerchner. dSprites: Disentanglement testing sprites dataset, 2017. 
URL https://github.com/deepmind/dsprites-dataset/.
'''

import numpy as np
import random

import torch
from filelock import FileLock
from torch.utils.data import Dataset

# reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

NUM_WORKERS = 0

class Torch_Dataset(Dataset):
    def __init__(self, X, Z, O, f_structure, Y):
        self.X = X
        self.Z = Z
        self.O = O
        self.f_structure = f_structure
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.Z[idx, :], self.O[idx, :], self.f_structure[idx, :], self.Y[idx, :]
    
class Base_Dataset():
    def __init__(self, 
                 seed, 
                 X, Z, O, f_structure, Y,
                 train_size, stage1_size, test_size, batch_size
                 ):
        
        self.X = X
        self.Z = Z
        self.O = O
        self.f_structure = f_structure
        self.Y = Y

        self.batch_size = batch_size
        
        # stage 1
        self.stage1_idx = np.arange(stage1_size)
        # stage 2
        self.train_idx = np.arange(train_size)
        # test
        self.test_idx = np.arange(stage1_size, stage1_size + test_size)

        assert len(set(self.stage1_idx).intersection(set(self.test_idx))) == 0
        assert train_size <= stage1_size

        # for stage 1, we only keep the y corresponding to train_idx
        # and set those corresponding to stage1_idx to be nan values
          # get the difference between stage1_idx and train_idx
        idx_diff = np.setdiff1d(self.stage1_idx, self.train_idx)
        self.Y[idx_diff, :] = float('nan')

        # datasets
        self.stage1_dataset = Torch_Dataset(
                            X=self.X[self.stage1_idx, :], 
                            Z=self.Z[self.stage1_idx, :],
                            O=self.O[self.stage1_idx, :],
                            f_structure=self.f_structure[self.stage1_idx, :],
                            Y=self.Y[self.stage1_idx, :]
                            )
        self.stage2_dataset = Torch_Dataset(
                            X=self.X[self.train_idx, :], 
                            Z=self.Z[self.train_idx, :],
                            O=self.O[self.train_idx, :],
                            f_structure=self.f_structure[self.train_idx, :],
                            Y=self.Y[self.train_idx, :]
                            )

    
    def get_train_loader(self, mode='stage1', use_validation=False):
        '''
        Sample a batch of data from the dataset
        return dataloader
        '''
        if mode == 'stage1':
            dataset = self.stage1_dataset
        elif mode == 'stage2':
            dataset = self.stage2_dataset
        else:
            raise NotImplementedError
        # if validation set is needed, split the dataset into train and validation (90-10)
        if use_validation:
            train_size = int(0.9 * len(dataset))
            validation_size = len(dataset) - train_size
            train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size], generator=g)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
            validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=validation_size, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
            return train_loader, validation_loader
        else:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
            return train_loader, None
            
    def get_samples(self, mode='train_all'):
        '''
        Sample a batch of data from the dataset
        return dataloader
        '''
        if mode == 'test':
            idx = self.test_idx
        elif mode == 'train_all':
            idx = self.train_idx
        else:
            raise NotImplementedError
        # return X, Z, O, f, Y. 
        return self.X[idx, :], self.Z[idx, :], self.O[idx, :], self.f_structure[idx, :], self.Y[idx, :]
    

    def get_dims(self):
        '''
        Get the dimension of the dataset
        '''
        return {'x_dim': self.X.shape[1], 
                'z_dim': self.Z.shape[1], 
                'o_dim': self.O.shape[1], 
                'y_dim': self.Y.shape[1]}
    
    def get_train_size(self):
        '''
        Get the training size
        '''
        return self.train_idx.shape[0]
    

    




