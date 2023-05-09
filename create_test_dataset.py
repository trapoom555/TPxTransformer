#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class CreateTestDataset:
    
# =============================================================================
#     Description: This Class aims to return PyTorch Dataloader Object
#     that contains Reverse mapping of inverse sum data to test 
#     Transformer Model as Seq2Seq manner
#     0 : padding
#     1 : <SOS>
#     2 : <EOS>
#     
#     For Example if we choose max_num to be 7 and max_len to be 7:
#     the number will be 3-7 (0, 1, 2 are reserved for padding, SOS, EOS)
#     X : [1, 7, 6, 5, 2, 0, 0]
#     (tmp : [1, 3, 4, 5, 2, 0, 0]) -> map token (max(max_num) to min(3))
#     Y : [1, 5, 4, 3, 2, 0, 0]     -> map order (reverse)
# =============================================================================
    
    def __init__(self, N_train: int, N_test: int, max_len: int, max_num: int, batch_size: int):
        assert max_len > 5
        assert max_num > 2
        
        self.N_train = N_train
        self.N_test = N_test
        self.max_len = max_len
        self.max_num = max_num
        self.batch_size = batch_size
        
        self.Xs_train = []
        self.ys_train = []
        
        self.Xs_test = []
        self.ys_test = []
    
    def generate_sample(self):
        data_len = np.random.randint(low = 3, high = self.max_len-1)
        
        # For no padding
        # data_len = self.max_len-2
        
        # generate x
        x = np.random.randint(low = 3, high = self.max_num+1, size = data_len)
        
        # reverse number
        y = self.max_num + 3 - x
        
        # reverse order
        y = y[::-1]
        
        # add SOS, EOS token
        x = np.insert(x, 0, 1)
        y = np.insert(y, 0, 1)
        
        x = np.append(x, 2)
        y = np.append(y, 2)
        
        # pad
        x = np.pad(x, (0, self.max_len - len(x)), 'constant', constant_values=0)
        y = np.pad(y, (0, self.max_len - len(y)), 'constant', constant_values=0)
        
        return x, y
    
    def generate_torch_dataloader(self):
        
        for i in range(self.N_train):
            x, y = self.generate_sample()
            
            self.Xs_train.append(x)
            self.ys_train.append(y)
        
        for i in range(self.N_test):
            x, y = self.generate_sample()
            
            self.Xs_test.append(x)
            self.ys_test.append(y)
        
        # transform to torch tensor
        tensor_x_train = torch.Tensor(self.Xs_train).to(torch.int64)
        tensor_y_train = torch.Tensor(self.ys_train).to(torch.int64)
        
        tensor_x_test = torch.Tensor(self.Xs_test).to(torch.int64)
        tensor_y_test = torch.Tensor(self.ys_test).to(torch.int64)
        
        # create PyTorch dataset
        dataset_train = TensorDataset(tensor_x_train,tensor_y_train)
        dataset_test = TensorDataset(tensor_x_test,tensor_y_test)
        
        # create PyTorch dataloader
        dataloader_train = DataLoader(dataset_train, batch_size = self.batch_size, shuffle = True)
        dataloader_test = DataLoader(dataset_test, batch_size = self.batch_size, shuffle = True)
        
        return dataloader_train, dataloader_test
        