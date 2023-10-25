import random
import torch
import unittest

from tests.envy import flush_tmp_dir

from torchness.types import TNS
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from torchness.tbwr import TBwr

BASE_DIR = f'{flush_tmp_dir()}/comoneural/zeroes'

# returns tensor of 0 with randomly set N elements to 1
def get_vector(
        width: int=      10,
        n: int=          1,
        rand_one: float= 0.01
) -> TNS:
    v = torch.zeros(width).to(int)
    for _ in range(n):
        if random.random() < rand_one:
            v[random.randrange(width)] = 1
    return v


class TestZeroesProcessor(unittest.TestCase):

    def setUp(self) -> None:
        flush_tmp_dir()

    def test_base(self):

        zepro = ZeroesProcessor(
            intervals=  (10, 50, 100),
            tbwr=       TBwr(logdir=BASE_DIR))

        for _ in range(10000):

            v = get_vector(width=10, n=2, rand_one=0.1)

            # very often change fixed positions to 1
            if random.random() < 0.95: v[0] = 1
            if random.random() < 0.95: v[1] = 1
            if random.random() < 0.95: v[2] = 1

            nane = zepro.process(zeroes=v)
            if 100 in nane:
                print(nane)