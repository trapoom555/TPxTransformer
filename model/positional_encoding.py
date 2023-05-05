#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from torch import Tensor, nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        
# =============================================================================
#         Description: Add Positional Encoding to Word Embedding Vector
#         Input: [batch_size, seq_len, d_model]
#         Output: [batch_size, seq_len, d_model]
# =============================================================================
        
        # d_model = embedding size
        # max_len (use to create the buffer)
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # [[0], [1], ... , [5000]]
        position = torch.arange(max_len).unsqueeze(1)
        
        # equivalent to 1/(10000**(2i/d_model))
        # exponential fn is faster than original fn
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) 
        pe = torch.zeros(max_len, 1, d_model)

        pe[:, 0, ::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # store intermediate computation in buffer to be reused in forward path
        self.register_buffer('pe', pe) 

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)] # plus positional encoding !
        return self.dropout(x)

