from pathlib import Path

import numpy as np
import pytest
from pypaq.lipytools.files import prep_folder

from torchness.models.text_embbeder import TextEMB, TextEMB_MOTorch

TMP_DIR = Path(__file__).parent / '_tmp_text_embbeder'
MOTORCH_DIR = TMP_DIR / 'motorch'


@pytest.fixture(autouse=True)
def tmp_dir():
    prep_folder(MOTORCH_DIR, flush_non_empty=True)
    TextEMB_MOTorch.SAVE_TOPDIR = str(MOTORCH_DIR)


def test_base_init():
    mt = TextEMB_MOTorch(module_type=TextEMB)
    print(mt.width)


def test_reinit():
    mt = TextEMB_MOTorch(module_type=TextEMB)
    print(mt)
    mt.save()
    mr = TextEMB_MOTorch(module_type=TextEMB)


def test_tokenize():
    mt = TextEMB_MOTorch(module_type=TextEMB)
    tokens = mt.get_tokens('This is Sparta')
    assert type(tokens) is list and type(tokens[0]) is str
    print(tokens)
    tokens = mt.get_tokens(['This is Sparta', 'No, it is not.'])
    assert type(tokens) is list and type(tokens[0][0]) is str
    print(tokens)


def test_encode():
    mt = TextEMB_MOTorch(module_type=TextEMB)
    emb = mt.get_embeddings(['This is Sparta', 'No, it is not.'])
    assert type(emb) is np.ndarray and emb.shape[0] == 2
    print(emb.shape)
