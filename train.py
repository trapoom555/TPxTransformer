#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("./model")

import torch
from create_test_dataset import CreateTestDataset
from model.transformer import Transformer
from torch import nn

# =============================================================================
# parameters zone
# =============================================================================

max_len = 30
ntoken = 15
Nx= 1
d_model = 120
num_heads = 1
d_ff = 240

N_train = 64*700
N_test = 64*300
batch_size = 64

# =============================================================================
# check available computing device
# =============================================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# =============================================================================
# define model and load it to computing device
# =============================================================================

model = Transformer(ntoken, Nx, d_model, num_heads, max_len = max_len, d_ff=d_ff).to(device)

# =============================================================================
# define dataset and get train, test dataloader
# =============================================================================

c = CreateTestDataset(N_train, N_test, 10, max_num = ntoken - 1, batch_size = batch_size)
train_dataloader, test_dataloader = c.generate_torch_dataloader()

# =============================================================================
# training loop
# =============================================================================

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Shift target left to be another model input
        X, y = X.to(device), y.to(device)
        y_input = y[:, :-1]
        y_output = y[:, 1:]
        
        # Compute prediction and loss
        pred = model(X, y_input)
        pred = pred.permute(0, 2, 1)
        loss = loss_fn(pred, y_output)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_input = y[:, :-1]
            y_output = y[:, 1:]
            
            pred = model(X, y_input)
            pred = pred.permute(0, 2, 1)
            test_loss += loss_fn(pred, y_output).item()
            correct += (pred.argmax(1) == y_output).type(torch.float).sum().item() / pred.shape[2]
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")

# =============================================================================
# sample prediction
# =============================================================================

with torch.no_grad():
    x = torch.Tensor([[1, 14, 5, 9, 7, 14, 3, 7, 3, 2]]).to(torch.int64).to(device)
    y = torch.Tensor([[1, 14]]).to(torch.int64).to(device)
    pred = model(x, y)
    next_token = pred.argmax(2)[:, -1]
    print(next_token)

