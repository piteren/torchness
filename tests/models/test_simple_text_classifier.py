from pathlib import Path

import numpy as np
import pytest
from pypaq.lipytools.files import prep_folder

from torchness.models.simple_text_classifier import STextCSF, STextCSF_MOTorch

TMP_DIR = Path(__file__).parent / '_tmp_simple_text_classifier'
MOTORCH_DIR = TMP_DIR / 'motorch'


@pytest.fixture(autouse=True)
def tmp_dir():
    prep_folder(MOTORCH_DIR, flush_non_empty=True)
    STextCSF_MOTorch.SAVE_TOPDIR = str(MOTORCH_DIR)


def test_base():
    mt = STextCSF_MOTorch(module_type=STextCSF)
    print(mt)


def test_embeddings():
    mt = STextCSF_MOTorch()
    emb = mt.get_embeddings(['This is Sparta', 'No, it is not.'])
    assert type(emb) is np.ndarray and emb.shape[0] == 2
    print(emb.shape)


def test_probs():
    mt = STextCSF_MOTorch()
    probs = mt.get_probs(['This is Sparta', 'No, it is not.'])
    assert type(probs) is np.ndarray and probs.shape == (2, 2)
    print(probs.shape)


def test_probsL():
    mt = STextCSF_MOTorch()
    probs = mt.get_probsL([
        ['This is Sparta', 'No, it is not.'],
        ['This is Sparta', 'No, it is not.', 'This is Sparta']])
    assert type(probs) is list and probs[1].shape == (3, 2)
    print(probs[1].shape)
