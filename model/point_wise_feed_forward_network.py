#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import Tensor, nn

class PointWiseFeedForwardNetwork(nn.Module):

# =============================================================================
#     Description: Transform Data by Sparse-Auto Encoder
#     input: Tensor [batch, seq_len, d_model]
#     output: Tensor [batch, seq_len, d_model]
# =============================================================================

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        
        self.nn1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        
        self.nn2 = nn.Linear(d_ff, d_model)
        
        self._init_nn_weights()

    def _init_nn_weights(self):
        # according to "Attention is all you need"
        nn.init.xavier_uniform_(self.nn1.weight)
        self.nn1.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.nn2.weight)
        self.nn2.bias.data.fill_(0)
        
    def forward(self, x: Tensor):
        x = self.nn1(x)
        x = self.relu(x)
        
        x = self.nn2(x)
        
        return x
        

