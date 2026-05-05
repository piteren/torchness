from pathlib import Path
import pytest
import torch
from pypaq.lipytools.files import prep_folder

from torchness.ckpt import mrg_ckpts, ckpt_nfo
from torchness.layers import LayDense
from torchness.motorch import Module, MOTorch

TMP_DIR = Path(__file__).parent / '_tmp_ckpt'
MOTORCH_DIR = TMP_DIR / 'motorch'


@pytest.fixture(autouse=True)
def tmp_dir():
    prep_folder(MOTORCH_DIR, flush_non_empty=True)
    MOTorch.SAVE_TOPDIR = str(MOTORCH_DIR)


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
        self.in_drop_lay = torch.nn.Dropout(p=in_drop) if in_drop > 0 else None
        self.lin = LayDense(in_features=in_shape, out_features=out_shape)
        self.loss_func = loss_func

    def forward(self, inp) -> dict:
        if self.in_drop_lay is not None: inp = self.in_drop_lay(inp)
        logits = self.lin(inp)
        return {'logits': logits}

    def loss(self, inp, lbl) -> dict:
        out = self(inp)
        out['loss'] = self.loss_func(out['logits'], lbl)
        out['acc'] = self.accuracy(out['logits'], lbl)
        return out


def test_mrg_ckpts():
    model = MOTorch(name='modA', module_type=LinModel, in_drop=0.1, device=None)
    model.save()
    model = MOTorch(name='modB', module_type=LinModel, in_drop=0.1, device=-1)
    model.save()

    mrg_ckpts(
        ckptA=  f'{MOTORCH_DIR}/modA/modA.pt',
        ckptB=  f'{MOTORCH_DIR}/modB/modB.pt',
        ckptM=  f'{MOTORCH_DIR}/ckptM.pt',
        ratio=  0.4,
        noise=  0.1)

    ckpt_nfo(f'{MOTORCH_DIR}/ckptM.pt')
