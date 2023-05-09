#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn, Tensor
import torch

def masked_NLLLoss(pred: Tensor, target: Tensor, masked_idx: int = 0):

# =============================================================================
#         Description: Masked Negative Log Likelihood Loss
#         pred: [batch, ntoken, de_seq_len]
#         target: [batch, de_seq_len]
#         Output: [batch_size, seq_len, d_model]
# =============================================================================

    nnl_loss = nn.NLLLoss()
    
    l = nnl_loss(pred, target)
    if l.shape != torch.Size([]):
        padding_mask = target != masked_idx
    
        l *= padding_mask
        
    return l

