#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn, Tensor

from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder
import torch.nn.functional as F

class Transformer(nn.Module):
    
# =============================================================================
#     Description: Full Transformer Model
#     input: Tensor [batch, seq_len]
#     output: Tensor [batch, seq_len, n_token]
# =============================================================================

    def __init__(self, ntoken: int, Nx: int, d_model: int, num_heads: int, dropout: float = 0.1, d_ff: int = 2048, max_len: int = 5000):
        super().__init__()    
        
        self.input_embed_map = nn.Embedding(ntoken, d_model, padding_idx=0)
        self.output_embed_map = nn.Embedding(ntoken, d_model, padding_idx=0)
        
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        self.encoder = Encoder(Nx, d_model, num_heads, dropout, d_ff)
        
        self.decoder = Decoder(Nx, d_model, num_heads, dropout, d_ff)
        
        self.output_proj = nn.Linear(d_model, ntoken)
        
    
    def forward(self, x: Tensor, y: Tensor):
        
        x = self.input_embed_map(x)
        x = self.positional_encoding(x)
        encoder_embed = self.encoder(x)
        
        y = self.output_embed_map(y)
        decoder_embed = self.decoder(y, encoder_embed)
        
        out = self.output_proj(decoder_embed)
        out = F.softmax(out, dim=-1)
        
        return out
