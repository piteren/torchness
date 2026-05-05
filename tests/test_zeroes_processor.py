import random
from pathlib import Path

import pytest
import torch
from pypaq.lipytools.files import prep_folder

from torchness.base import TNS
from torchness.tbwr import TBwr
from torchness.zeroes_processor import ZeroesProcessor

TMP_DIR = Path(__file__).parent / '_tmp_zeroes_processor'


@pytest.fixture(autouse=True)
def tmp_dir():
    prep_folder(TMP_DIR, flush_non_empty=True)


def get_vector(
        width: int = 10,
        n: int = 1,
        rand_one: float = 0.01,
) -> TNS:
    v = torch.zeros(width).to(int)
    for _ in range(n):
        if random.random() < rand_one:
            v[random.randrange(width)] = 1
    return v


def test_base():
    zepro = ZeroesProcessor(
        intervals=  (10, 50, 100),
        tbwr=       TBwr(logdir=str(TMP_DIR)))

    for _ in range(10000):
        v = get_vector(width=10, n=2, rand_one=0.1)
        if random.random() < 0.95: v[0] = 1
        if random.random() < 0.95: v[1] = 1
        if random.random() < 0.95: v[2] = 1

        nane = zepro.process(zeroes=v)
        if 100 in nane:
            print(nane)
