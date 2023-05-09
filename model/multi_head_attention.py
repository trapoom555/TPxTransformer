#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from torch import Tensor, nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

class ScaledDotProduct(nn.Module):

# =============================================================================
#     Description: Calculate ScaledDotProduct from Q, K, V
#     q: Tensor [batch, num_heads, seq_len_1, d_k]
#     k: Tensor [batch, num_heads, seq_len_2, d_k]
#     v: Tensor [batch, num_heads, seq_len_2, d_k]
#     output: Tensor [batch, num_heads, seq_len_1, d_k]
# =============================================================================
    
    def __init__(self, causal_mask: bool):
        super().__init__()
        
        self.causal_mask = causal_mask
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        d_k = q.shape[-1] # head dimension
        dot_product = q @ k.transpose(-2, -1) # [batch, num_heads, seq_len_1, seq_len_2]
        scaled_dot_product = dot_product / math.sqrt(d_k)
        
        # lower triangular mask (prevent left-ward information flow)
        if self.causal_mask == True:
            mask = torch.tril(torch.ones_like(scaled_dot_product))
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, -9e15)
    
        attention_weight = F.softmax(scaled_dot_product, dim=-1)
        #plt.imshow(attention_weight[0, 0, :, :].detach().cpu().numpy(), cmap='magma')
        #plt.show()
        weighted_v = attention_weight @ v # [batch, num_heads, seq_len_1, d_k]
        return weighted_v
        
        

class MultiHeadSelfAttention(nn.Module):
    
# =============================================================================
#     Description: Project Word Embeddings to Q, K, V then do MultiHeadSelfAttention
#     input: Tensor [batch, seq_len, d_model]
#     output: Tensor [batch, seq_len, d_model]
# =============================================================================
    
    
    def __init__(self, input_dim: int, d_model: int, num_heads: int, causal_mask : bool = False):
        super().__init__()
        
        try:
            assert d_model % num_heads == 0
        except:
            raise Exception("d_model % num_heads != 0")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # because they will be later concat
        
        # scaled_dot_product
        self.scaled_dot_product = ScaledDotProduct(causal_mask)
        
        # NN to project embedding to Q vector
        self.q_proj = nn.Linear(input_dim, d_model)
        
        # NN to project embedding to K vector
        self.k_proj = nn.Linear(input_dim, d_model)
        
        # NN to project embedding to V vector
        self.v_proj = nn.Linear(input_dim, d_model)
        
        # output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._init_nn_weights()
    
    def _init_nn_weights(self):
        # according to "Attention is all you need"
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.bias.data.fill_(0)
    
    def forward(self, x: Tensor):
        batch_size, seq_len, _ = x.shape
        
        # project embedding to Q, K, V vector
        q = self.q_proj(x) # [batch, seq_len, d_model]
        k = self.k_proj(x) # [batch, seq_len, d_model]
        v = self.v_proj(x) # [batch, seq_len, d_model]
        
        # reshape
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # permute (prepare to scaled_dot_product)
        q = q.permute(0, 2, 1, 3) # [batch, num_heads, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3) # [batch, num_heads, seq_len, head_dim]
        v = v.permute(0, 2, 1, 3) # [batch, num_heads, seq_len, head_dim]
        
        # calculate scaled_dot_product
        weighted_v = self.scaled_dot_product(q, k, v) # [batch, num_heads, seq_len, head_dim]
        
        # concat each head
        weighted_v = weighted_v.permute(0, 2, 1, 3) # [batch, seq_len, num_heads, head_dim]
        concat_weighted_v = weighted_v.reshape(batch_size, seq_len, self.d_model)
        
        # project concat_weighted_v to output matrix
        output = self.output_proj(concat_weighted_v)
        
        return output


class MultiHeadEncoderDecoderAttention(nn.Module):
    
# =============================================================================
#     Description: Project Word Embeddings to Q, K, V then do MultiHeadEncoderDecoderAttention
#     en: Tensor [batch, seq_len, d_model]
#     de: Tensor [batch, seq_len, d_model]
#     output: Tensor [batch, seq_len, d_model]
# =============================================================================
    
    
    def __init__(self, input_dim: int, d_model: int, num_heads: int, causal_mask : bool = False):
        super().__init__()
        
        try:
            assert d_model % num_heads == 0
        except:
            raise Exception("d_model % num_heads != 0")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # because they will be later concat
        
        # scaled_dot_product
        self.scaled_dot_product = ScaledDotProduct(causal_mask)
        
        # NN to project embedding to K vector from encoder
        self.k_proj = nn.Linear(input_dim, d_model)
        
        # NN to project embedding to V vector from encoder
        self.v_proj = nn.Linear(input_dim, d_model)
        
        # NN to project embedding to Q vector from decoder
        self.q_proj = nn.Linear(input_dim, d_model)
        
        # output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._init_nn_weights()
    
    def _init_nn_weights(self):
        # according to "Attention is all you need"
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.bias.data.fill_(0)

    def forward(self, en: Tensor, de: Tensor):
        batch_size, en_seq_len, _ = en.shape
        batch_size, de_seq_len, _ = de.shape
        
        # project embedding from encoder to K vector
        k = self.k_proj(en) # [batch, seq_len, d_model]
        
        # project embedding from encoder to V vector
        v = self.v_proj(en) # [batch, seq_len, d_model]
        
        # project embedding from decoder to Q vector
        q = self.q_proj(de) # [batch, seq_len, d_model]
        
        # reshape
        q = q.reshape(batch_size, de_seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, en_seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, en_seq_len, self.num_heads, self.head_dim)
        
        # permute (prepare to scaled_dot_product)
        q = q.permute(0, 2, 1, 3) # [batch, num_heads, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3) # [batch, num_heads, seq_len, head_dim]
        v = v.permute(0, 2, 1, 3) # [batch, num_heads, seq_len, head_dim]

        
        # calculate scaled_dot_product
        weighted_v = self.scaled_dot_product(q, k, v) # [batch, num_heads, seq_len, head_dim]
        
        # concat each head
        weighted_v = weighted_v.permute(0, 2, 1, 3) # [batch, seq_len, num_heads, head_dim]
        concat_weighted_v = weighted_v.reshape(batch_size, de_seq_len, self.d_model)
        
        # project concat_weighted_v to output matrix
        output = self.output_proj(concat_weighted_v)
        
        return output

        
        
        
        
        
        
        
    