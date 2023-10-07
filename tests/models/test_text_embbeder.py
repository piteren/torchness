import unittest

from torchness.models.text_embbeder import TextEMB
from torchness.motorch import MOTorch

from tests.envy import flush_tmp_dir

MOTORCH_DIR = f'{flush_tmp_dir()}/motorch'
MOTorch.SAVE_TOPDIR = MOTORCH_DIR


class TestSFeatsCSF(unittest.TestCase):

    def setUp(self) -> None:
        flush_tmp_dir()

    def test_base(self):
        mt = MOTorch(module_type=TextEMB)