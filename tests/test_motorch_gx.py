from pypaq.mpython.mpdecor import proc_wait
from pypaq.lipytools.pylogger import get_pylogger
import torch
import unittest

from torchness.motorch import MOTorch, Module, MOTorchException
from torchness.layers import LayDense

from tests.envy import flush_tmp_dir

MOTORCH_DIR = f'{flush_tmp_dir()}/motorch_gx'
MOTorch.SAVE_TOPDIR = MOTORCH_DIR


class LinModel(Module):

    def __init__(
            self,
            in_drop: float,
            in_shape=   784,
            out_shape=  10,
            loss_func=  torch.nn.functional.cross_entropy,
            device=     None,
            dtype=      None,
            seed=       121,
            **kwargs,
    ):

        Module.__init__(self, **kwargs)

        self.in_drop_lay = torch.nn.Dropout(p=in_drop) if in_drop>0 else None
        self.lin = LayDense(in_features=in_shape, out_features=out_shape)
        self.loss_func = loss_func

        self.logger.debug('LinModel initialized!')

    def forward(self, inp) -> dict:
        if self.in_drop_lay is not None: inp = self.in_drop_lay(inp)
        logits = self.lin(inp)
        return {'logits': logits}

    def loss(self, inp, lbl) -> dict:
        out = self(inp)
        out['loss'] = self.loss_func(out['logits'], lbl)
        out['acc'] = self.accuracy(out['logits'], lbl)  # using baseline
        return out


logger = get_pylogger(name='test_motorch', level=20)


class TestMOTorch_GX(unittest.TestCase):


    def setUp(self) -> None:
        flush_tmp_dir()


    def test_gx_ckpt(self):

        name_A = 'modA'
        name_B = 'modB'

        # needs to create in separate process
        @proc_wait
        def create():
            model = MOTorch(
                module_type=    LinModel,
                name=           name_A,
                seed=           121,
                in_drop=        0.1,
                device=         None,
                logger=         logger)
            model.save()

            model = MOTorch(
                module_type=    LinModel,
                name=           name_B,
                seed=           121,
                in_drop=        0.1,
                device=         None,
                logger=         logger)
            model.save()

        create()

        MOTorch.gx_ckpt(
            name_A=         name_A,
            name_B=         name_B,
            name_child=     f'{name_A}_GXed')


    def test_gx_saved(self):

        name_C = 'modC'
        name_D = 'modD'

        # needs to create in separate process
        @proc_wait
        def create():
            model = MOTorch(
                module_type=    LinModel,
                name=           name_C,
                seed=           121,
                in_drop=        0.1,
                device=         None,
                logger=         logger)
            model.save()

            model = MOTorch(
                module_type=    LinModel,
                name=           name_D,
                seed=           121,
                in_drop=        0.1,
                device=         None,
                logger=         logger)
            model.save()

        create()

        MOTorch.gx_saved(
            name_parent_main=   name_C,
            name_parent_scnd=   name_D,
            name_child=         f'{name_C}_GXed')
