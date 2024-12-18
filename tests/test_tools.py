import torch
import unittest

from torchness.tools import select_with_indices


class TestTools(unittest.TestCase):

    def test_select_with_indices(self):

        source = torch.rand(4,3)
        print(source)
        indices = [1,0,2,1]
        indices = torch.tensor(indices)
        print(indices)
        swi = select_with_indices(source,indices)
        print(swi)

        _swi = source[range(len(indices)), indices]
        print(_swi)

        self.assertTrue(torch.equal(swi,_swi))





