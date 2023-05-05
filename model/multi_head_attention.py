#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from torch import Tensor, nn
import torch.nn.functional as F
import torch

class ScaledDotProduct(nn.Module):

# =============================================================================
#     Description: Calculate ScaledDotProduct from Q, K, V
#     q: Tensor [batch, num_heads, seq_len_1, d_k]
#     k: Tensor [batch, num_heads, seq_len_2, d_k]
#     v: Tensor [batch, num_heads, seq_len_2, d_k]
#     output: Tensor [batch, num_heads, seq_len_1, d_k]
# =============================================================================
    
    def __init__(self):
        super().__init__()
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask):
        d_k = q.shape[-1] # head dimension
        dot_product = q @ k.transpose(-2, -1) # [batch, num_heads, seq_len_1, seq_len_2]
        scaled_dot_product = dot_product / math.sqrt(d_k)
        
        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, -9e15)
    
        attention_weight = F.softmax(scaled_dot_product, dim=-1)
        weighted_v = attention_weight @ v # [batch, num_heads, seq_len_1, d_k]
        return weighted_v
        
        

class MultiHeadSelfAttention(nn.Module):
    
# =============================================================================
#     Description: Project Word Embeddings to Q, K, V then do MultiHeadSelfAttention
#     input: Tensor [batch, seq_len, d_model]
#     output: Tensor [batch, seq_len, d_model]
# =============================================================================
    
    
    def __init__(self, input_dim: int, d_model: int, num_heads: int):
        super().__init__()
        
        try:
            assert d_model % num_heads == 0
        except:
            raise Exception("d_model % num_heads != 0")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # because they will be later concat
        
        # scaled_dot_product
        self.scaled_dot_product = ScaledDotProduct()
        
        # NN to project embedding to QKV tensor stack
        self.qkv_proj = nn.Linear(input_dim, 3 * d_model)
        
        # output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._init_nn_weights()
    
    def _init_nn_weights(self):
        # according to "Attention is all you need"
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.bias.data.fill_(0)
    
    def forward(self, x: Tensor, mask):
        batch_size, seq_len, _ = x.shape
        
        # project embedding to QKV tensor stack
        qkv_stack = self.qkv_proj(x) # [batch, seq_len, 3 * d_model]
        
        # reshape
        qkv_stack = qkv_stack.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        
        # permute (prepare to scaled_dot_product)
        qkv_stack = qkv_stack.permute(0, 2, 1, 3) # [batch, num_heads, seq_len, 3*head_dim]
        
        # chunk !
        q, k, v = qkv_stack.chunk(3, dim=-1) # [batch, num_heads, seq_len, head_dim] each
        
        # calculate scaled_dot_product
        weighted_v = self.scaled_dot_product(q, k, v, mask) # [batch, num_heads, seq_len, head_dim]
        
        # concat each head
        weighted_v = weighted_v.permute(0, 2, 1, 3) # [batch, seq_len, num_heads, head_dim]
        concat_weighted_v = weighted_v.reshape(batch_size, seq_len, self.d_model)
        
        # project concat_weighted_v to output matrix
        output = self.output_proj(concat_weighted_v)
        
        return output


class MultiHeadEncoderDecoderAttention(nn.Module):
    
# =============================================================================
#     Description: Project Word Embeddings to Q, K, V then do MultiHeadEncoderDecoderAttention
#     en: Tensor [batch, seq_len, word_embed_dim]
#     de: Tensor [batch, seq_len, word_embed_dim]
#     output: Tensor [batch, seq_len, d_model]
# =============================================================================
    
    
    def __init__(self, input_dim: int, d_model: int, num_heads: int):
        super().__init__()
        
        try:
            assert d_model % num_heads == 0
        except:
            raise Exception("d_model % num_heads != 0")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # because they will be later concat
        
        # scaled_dot_product
        self.scaled_dot_product = ScaledDotProduct()
        
        # NN to project embedding to KV tensor stack from encoder
        self.kv_proj = nn.Linear(input_dim, 2 * d_model)
        
        # NN to project embedding to Q tensor from encoder
        self.q_proj = nn.Linear(input_dim, d_model)
        
        # output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._init_nn_weights()
    
    def _init_nn_weights(self):
        # according to "Attention is all you need"
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.bias.data.fill_(0)
    
    def forward(self, en: Tensor, de: Tensor, mask):
        batch_size, seq_len, _ = en.shape
        
        # project embedding from encoder to KV tensor stack
        kv_stack = self.kv_proj(en) # [batch, seq_len, 2 * d_model]
        
        # project embedding from encoder to KV tensor stack
        q = self.q_proj(de) # [batch, seq_len, d_model]
        
        # build QKV stack
        qkv_stack = torch.cat((q, kv_stack), dim=-1)
        
        # reshape
        qkv_stack = qkv_stack.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        
        # permute (prepare to scaled_dot_product)
        qkv_stack = qkv_stack.permute(0, 2, 1, 3) # [batch, num_heads, seq_len, 3*head_dim]
        
        # chunk !
        q, k, v = qkv_stack.chunk(3, dim=-1) # [batch, num_heads, seq_len, head_dim] each
        
        # calculate scaled_dot_product
        weighted_v = self.scaled_dot_product(q, k, v, mask) # [batch, num_heads, seq_len, head_dim]
        
        # concat each head
        weighted_v = weighted_v.permute(0, 2, 1, 3) # [batch, seq_len, num_heads, head_dim]
        concat_weighted_v = weighted_v.reshape(batch_size, seq_len, self.d_model)
        
        # project concat_weighted_v to output matrix
        output = self.output_proj(concat_weighted_v)
        
        return output

        
        
        
        
        
        
        
    