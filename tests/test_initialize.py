import torch

from torchness.initialize import my_initializer


def test_my_initializer():
    tns = torch.zeros(1000)
    my_initializer(tns, std=0.1)
    print(tns.numpy().std())
    assert 0.08 < tns.numpy().std() < 0.12
