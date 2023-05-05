#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import Tensor, nn
from multi_head_attention import MultiHeadSelfAttention
from point_wise_feed_forward_network import PointWiseFeedForwardNetwork

class EncoderBlock(nn.Module):

# =============================================================================
#     Description: One Encoder Block
#     input: Tensor [batch, seq_len, d_model]
#     output: Tensor [batch, seq_len, d_model]
# =============================================================================

    def __init__(self, d_model: int, num_heads: int, dropout: float, d_ff: int):
        super().__init__()
        
        self.self_attn = MultiHeadSelfAttention(d_model, d_model, num_heads)
        
        self.point_wise_feed_forward_network = PointWiseFeedForwardNetwork(d_model, d_ff)
        
        # normalize over last two dimensions
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor):
        # residual connection
        x = x + self.dropout(self.self_attn(x))
        x = self.norm1(x)
        
        x = x + self.dropout(self.point_wise_feed_forward_network(x))
        x = self.norm2(x)
        
        return x
        

class Encoder(nn.Module):

# =============================================================================
#     Description: Concat Nx EncoderBlocks
#     input: Tensor [batch, seq_len, d_model]
#     output: Tensor [batch, seq_len, d_model]
# =============================================================================

    def __init__(self, Nx: int, d_model: int, num_heads: int, dropout: float = 0.1, d_ff: int = 2048):
        super().__init__()
        
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, dropout, d_ff) for i in range(Nx)])
    
    def forward(self, x: Tensor):
        for e in self.encoder_blocks:
            x = e(x)
        
        return x
        
        
