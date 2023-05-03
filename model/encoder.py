#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import Tensor, nn
from multi_head_attention import MultiHeadSelfAttention

class EncoderBlock(nn.Module):

# =============================================================================
#     Description: One Encoder Block
#     input: Tensor [batch, seq_len, d_model]
#     output: Tensor [batch, seq_len, d_model]
# =============================================================================

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        
        self.self_attn = MultiHeadSelfAttention(d_model, d_model, num_heads)
        
        self.nn = nn.Linear(d_model, d_model)
        
        # normalize over last two dimensions
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_nn_weights()
    
    def _init_nn_weights(self):
        # according to "Attention is all you need"
        nn.init.xavier_uniform_(self.nn.weight)
        self.nn.bias.data.fill_(0)
    
    def forward(self, x: Tensor):
        # residual connection
        x = x + self.dropout(self.self_attn(x))
        x = self.norm1(x)
        
        x = x + self.dropout(self.nn(x))
        x = self.norm2(x)
        
        return x
        

class Encoder(nn.Module):

# =============================================================================
#     Description: Concat Nx EncoderBlocks
#     input: Tensor [batch, seq_len, d_model]
#     output: Tensor [batch, seq_len, d_model]
# =============================================================================

    def __init__(self, Nx: int, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, dropout) for i in range(Nx)])
    
    def forward(self, x: Tensor):
        for e in self.encoder_blocks:
            x = e(x)
        
        return x
        
        
