import torch

from torchness.scaled_LR import ScaledLR
from torchness.layers import LayDense


def test_base():
    lay = LayDense(2, 2)
    optimizer = torch.optim.AdamW(params=lay.parameters(), lr=1e-4)
    scaler = ScaledLR(optimizer=optimizer, loglevel=10)
    lr = scaler.get_lr()[0]
    print(lr)
    optimizer.step()
    scaler.step()
    lr = scaler.get_lr()[0]
    print(lr)
