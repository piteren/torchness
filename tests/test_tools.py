import torch

from torchness.tools import select_with_indices


def test_select_with_indices():
    source = torch.rand(4, 3)
    print(source)
    indices = [1, 0, 2, 1]
    indices = torch.tensor(indices)
    print(indices)
    swi = select_with_indices(source, indices)
    print(swi)

    _swi = source[range(len(indices)), indices]
    print(_swi)

    assert torch.equal(swi, _swi)
