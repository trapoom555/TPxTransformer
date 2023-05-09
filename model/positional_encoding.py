#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from torch import Tensor, nn
import torch

import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        
# =============================================================================
#         Description: Add Positional Encoding to Word Embedding Vector
#         Input: [batch_size, seq_len, d_model]
#         Output: [batch_size, seq_len, d_model]
# =============================================================================

        # max_len (use to create the buffer)
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # [max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)
        
        # equivalent to 1/(10000**(2i/d_model))
        # exponential fn is faster than power fn
        # [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) 
        
        pe = torch.zeros(max_len, d_model)
        
        pe[:, ::2] = torch.sin(position * div_term) # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term) # [max_len, d_model/2]

        plt.imshow(pe.detach().cpu().numpy(), cmap='magma')
        plt.show()
        # store intermediate computation in buffer to be reused in forward path
        self.register_buffer('pe', pe) 

    def forward(self, x: Tensor):
        # plus positional encoding !
        x = x + self.pe[:x.shape[1]] 
        return self.dropout(x)

