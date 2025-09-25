import unittest

from torchness.scaled_LR import ScaledLR


class TestScaledLR(unittest.TestCase):

    def test_base(self):
        scaler = ScaledLR()
        print(scaler)
        print(scaler.get_lr())