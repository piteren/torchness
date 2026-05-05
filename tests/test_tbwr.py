from pathlib import Path

import pytest
import torch
from pypaq.lipytools.files import prep_folder

from torchness.tbwr import TBwr

TMP_DIR = Path(__file__).parent / '_tmp_tbwr'


@pytest.fixture(autouse=True)
def tmp_dir():
    prep_folder(TMP_DIR, flush_non_empty=True)


def test_TBwr_values():
    tbwr = TBwr(logdir=str(TMP_DIR / 'val'))
    val = 1.7
    for ix in range(100):
        tbwr.add(value=val, tag='val', step=ix)
        val += 0.15


def test_TBwr_histogram():
    tbwr = TBwr(logdir=str(TMP_DIR / 'values_histogram'))
    vals = torch.rand(100)
    for ix in range(100):
        tbwr.add_histogram(values=vals, tag='vals_hist', step=ix)
        vals = vals / 1.05 + torch.rand(100) * 0.05
