from pathlib import Path

import pytest
import torch
from pypaq.lipytools.files import prep_folder

from torchness.models.simple_feats_classifier import SFeatsCSF
from torchness.motorch import MOTorch

TMP_DIR = Path(__file__).parent / '_tmp_simple_feats_classifier'
MOTORCH_DIR = TMP_DIR / 'motorch'


@pytest.fixture(autouse=True)
def tmp_dir():
    prep_folder(MOTORCH_DIR, flush_non_empty=True)
    MOTorch.SAVE_TOPDIR = str(MOTORCH_DIR)


def test_base():
    mt = MOTorch(module_type=SFeatsCSF, feats_width=128, num_classes=3)
    print(mt)


def test_fwd_bwd():
    mt = MOTorch(module_type=SFeatsCSF, feats_width=128, num_classes=3)

    inp = torch.rand(3, 128)
    out = mt(inp)
    print(out)

    labels = torch.tensor([0, 2, 1])
    print(labels)
    out = mt.loss(feats=inp, labels=labels)
    print(out)
