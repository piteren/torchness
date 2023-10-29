import numpy as np
from pypaq.lipytools.pylogger import get_pylogger
import unittest

from torchness.models.text_embbeder import TextEMB, TextEMB_MOTorch

from tests.envy import flush_tmp_dir

TextEMB_MOTorch.SAVE_TOPDIR = f'{flush_tmp_dir()}/motorch'

logger = get_pylogger(name='test_embedder', level=20)


class TestTextEMB(unittest.TestCase):

    def setUp(self) -> None:
        flush_tmp_dir()

    def test_base_init(self):
        mt = TextEMB_MOTorch(module_type=TextEMB)
        print(mt.width)

    def test_reinit(self):
        mt = TextEMB_MOTorch(module_type=TextEMB, logger=logger)
        print(mt)
        mt.save()
        mr = TextEMB_MOTorch(module_type=TextEMB, logger=logger)

    def test_tokenize(self):
        mt = TextEMB_MOTorch(module_type=TextEMB, logger=logger)
        tokens = mt.get_tokens('This is Sparta')
        self.assertTrue(type(tokens) is list and type(tokens[0]) is str)
        print(tokens)
        tokens = mt.get_tokens(['This is Sparta', 'No, it is not.'])
        self.assertTrue(type(tokens) is list and type(tokens[0][0]) is str)
        print(tokens)

    def test_encode(self):
        mt = TextEMB_MOTorch(module_type=TextEMB, logger=logger)
        emb = mt.get_embeddings(['This is Sparta', 'No, it is not.'])
        self.assertTrue(type(emb) is np.ndarray and emb.shape[0]==2)
        print(emb.shape)