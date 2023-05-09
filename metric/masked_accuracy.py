#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import Tensor

def masked_accuracy(pred: Tensor, target: Tensor, masked_idx: int = 0):
    
# =============================================================================
#         Description: Return average accuracy per seq in batch
#         pred: [batch, ntoken, de_seq_len]
#         target: [batch, de_seq_len]
#         Output: [batch_size, seq_len, d_model]
# =============================================================================
    
    padding_mask = target != masked_idx
    
    return ((pred.argmax(1) * padding_mask) == target).type(torch.float).sum().item() / pred.shape[2]
    

