#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import Tensor, nn
from multi_head_attention import MultiHeadSelfAttention, MultiHeadEncoderDecoderAttention
from point_wise_feed_forward_network import PointWiseFeedForwardNetwork

class DecoderBlock(nn.Module):

# =============================================================================
#     Description: One Decoder Block
#     input: Tensor [batch, seq_len, d_model]
#     output: Tensor [batch, seq_len, d_model]
# =============================================================================

    def __init__(self, d_model: int, num_heads: int, dropout: float, d_ff: int):
        super().__init__()
        
        self.self_attn = MultiHeadSelfAttention(d_model, d_model, num_heads, causal_mask=True)
        
        self.en_de_attn = MultiHeadEncoderDecoderAttention(d_model, d_model, num_heads)
        
        self.point_wise_feed_forward_network = PointWiseFeedForwardNetwork(d_model, d_ff)
        
        # normalize over last two dimensions
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, encoder_embed: Tensor):
        # residual connection
        x = x + self.dropout(self.self_attn(x))
        x = self.norm1(x)
        
        x = x + self.dropout(self.en_de_attn(encoder_embed, x))
        x = self.norm2(x)
        
        x = x + self.dropout(self.point_wise_feed_forward_network(x))
        x = self.norm3(x)
        
        return x
        

class Decoder(nn.Module):

# =============================================================================
#     Description: Concat Nx DecoderBlocks
#     input: Tensor [batch, seq_len, d_model]
#     output: Tensor [batch, seq_len, d_model]
# =============================================================================

    def __init__(self, Nx: int, d_model: int, num_heads: int, dropout: float, d_ff: int):
        super().__init__()
        
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, num_heads, dropout, d_ff) for i in range(Nx)])
    
    def forward(self, x: Tensor, encoder_embed: Tensor):
        for d in self.decoder_blocks:
            x = d(x, encoder_embed)
        
        return x
        
        

