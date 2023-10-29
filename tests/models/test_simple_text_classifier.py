import numpy as np
import unittest

from torchness.models.simple_text_classifier import STextCSF, STextCSF_MOTorch

from tests.envy import flush_tmp_dir

STextCSF_MOTorch.SAVE_TOPDIR = f'{flush_tmp_dir()}/motorch'


class TestSFeatsCSF(unittest.TestCase):

    def setUp(self) -> None:
        flush_tmp_dir()

    def test_base(self):
        mt = STextCSF_MOTorch(module_type=STextCSF)
        print(mt)

    def test_embeddings(self):
        mt = STextCSF_MOTorch()
        emb = mt.get_embeddings(['This is Sparta', 'No, it is not.'])
        self.assertTrue(type(emb) is np.ndarray and emb.shape[0] == 2)
        print(emb.shape)

    def test_probs(self):
        mt = STextCSF_MOTorch()
        probs = mt.get_probs(['This is Sparta', 'No, it is not.'])
        self.assertTrue(type(probs) is np.ndarray and probs.shape == (2,2))
        print(probs.shape)

    def test_probsL(self):
        mt = STextCSF_MOTorch()
        probs = mt.get_probsL([
            ['This is Sparta', 'No, it is not.'],
            ['This is Sparta', 'No, it is not.', 'This is Sparta']])
        self.assertTrue(type(probs) is list and probs[1].shape == (3,2))
        print(probs[1].shape)