import torch
import unittest

from torchness.base_elements import my_initializer, mrg_ckpts, ckpt_nfo, select_with_indices, reinforced_cross_entropy
from torchness.motorch import Module, MOTorch
from torchness.layers import LayDense

from tests.envy import flush_tmp_dir

MOTORCH_DIR = f'{flush_tmp_dir()}/motorch'
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


class TestBaseElements(unittest.TestCase):

    def test_my_initializer(self):
        tns = torch.zeros(1000)
        #print(tns)
        my_initializer(tns, std=0.1)
        #print(tns)
        print(tns.numpy().std())
        self.assertTrue(0.08 < tns.numpy().std() < 0.12)

    def test_select_with_indices(self):

        source = torch.rand(4,3)
        print(source)
        indices = [1,0,2,1]
        indices = torch.tensor(indices)
        print(indices)
        swi = select_with_indices(source,indices)
        print(swi)

        _swi = source[range(len(indices)), indices]
        print(_swi)

        self.assertTrue(torch.equal(swi,_swi))

    def test_reinforced_cross_entropy(self):

        logits = torch.rand(5,5)
        logits -= 0.5
        logits *=5
        print(f'logits: {logits}')
        probs = torch.softmax(logits, dim=-1)
        print(f'probs: {probs}, sum:{torch.sum(probs, dim=-1)}')

        target = torch.randint(5,(5,))
        lf = torch.nn.CrossEntropyLoss(reduction='none')
        ce_loss = lf(logits,target,)
        probs_target = probs[range(len(target)), target]
        print(f'probs_target: {probs_target}')
        print(f'cross entropy loss: {ce_loss}, ({-torch.log(probs_target)})')

        scale = (torch.rand(5) > 0.5).to(float)*2-1
        print(f'scale: {scale}')
        sce = reinforced_cross_entropy(
            labels=     target,
            scale=      scale,
            logits=     logits)
        print(f'sce: {sce}')



class TestCheckpoints(unittest.TestCase):

    def setUp(self) -> None:
        flush_tmp_dir()

    def test_mrg_ckpts(self):

        model = MOTorch(
            name=           'modA',
            module_type=    LinModel,
            in_drop=        0.1,
            device=         None)
        model.save()
        model = MOTorch(
            name=           'modB',
            module_type=    LinModel,
            in_drop=        0.1,
            device=         -1)
        model.save()

        mrg_ckpts(
            ckptA=  f'{MOTORCH_DIR}/modA/modA.pt',
            ckptB=  f'{MOTORCH_DIR}/modB/modB.pt',
            ckptM=  f'{MOTORCH_DIR}/ckptM.pt',
            ratio=  0.4,
            noise=  0.1,
        )

        ckpt_nfo(f'{MOTORCH_DIR}/ckptM.pt')


