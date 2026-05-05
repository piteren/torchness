from pathlib import Path

import numpy as np
import pytest
import torch
from pypaq.lipytools.files import prep_folder

from torchness.motorch import MOTorch, Module, MOTorchException
from torchness.layers import LayDense

TMP_DIR = Path(__file__).parent / '_tmp_motorch'
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
            seed=       121,
            **kwargs,
    ):
        Module.__init__(self, **kwargs)
        self.in_drop_lay = torch.nn.Dropout(p=in_drop) if in_drop > 0 else None
        self.lin = LayDense(in_features=in_shape, out_features=out_shape)
        self.loss_func = loss_func
        self.logger.debug('LinModel initialized!')

    def forward(self, inp) -> dict:
        if self.in_drop_lay is not None: inp = self.in_drop_lay(inp)
        logits = self.lin(inp)
        return {'logits': logits}

    def loss(self, inp, true) -> dict:
        out = self(inp)
        out['true'] = true
        out['loss'] = self.loss_func(out['logits'], true)
        return out


class LinModelOpt(LinModel):

    def get_optimizer_definition(self) -> tuple[type[torch.optim.Optimizer], dict]:
        return torch.optim.SGD, {'momentum': 0.666}


### init / build

def test_base_init():
    model = MOTorch(module_type=LinModel, in_drop=0.0)
    print(model)
    assert model.size == 7850
    assert not model.module.training
    assert type(model.module) is LinModel
    assert model.dtype == torch.float32


def test_init_raises():
    with pytest.raises(Exception):
        MOTorch()
    with pytest.raises(Exception):
        MOTorch(name='LinModel')
    with pytest.raises(Exception):
        MOTorch(module_type=LinModel)


def test_name_stamp():
    model = MOTorch(module_type=LinModel, in_drop=0.1)
    print(model['name'])
    assert model['name'] == 'LinModel_MOTorch'

    model = MOTorch(module_type=LinModel, name='LinTest', in_drop=0.1)
    print(model['name'])
    assert model['name'] == 'LinTest'

    model = MOTorch(module_type=LinModel, name_timestamp=True, in_drop=0.1)
    print(model['name'])
    assert model['name'] != 'MOTorch_LinModel'
    assert {d for d in '0123456789'} & set(model['name'])


def test_device():
    model = MOTorch(module_type=LinModel, device=-1, in_drop=0.0)
    dev = model.device
    print(dev)
    assert dev == ('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_loglevel():
    model = MOTorch(module_type=LinModel, device=-1, in_drop=0.0, loglevel=10)
    model.save()


### save / load / folder

def test_save_load():
    model = MOTorch(module_type=LinModel, loglevel=10, in_drop=0.1)
    assert model['seed'] == 121 and model['baseLR'] == 0.0003
    assert 'loss' not in model.get_managed_params()
    model.save()
    name = model.name

    print('\nsaved, now loading..')
    model = MOTorch(name=name, loglevel=10)
    assert model['in_drop'] == 0.1 and model['in_shape'] == 784


def test_read_only():
    model = MOTorch(module_type=LinModel, in_drop=0.1)
    model.save()
    name = model.name

    model = MOTorch(name=name, read_only=True)
    with pytest.raises(MOTorchException):
        model.save()


def test_save_load_full():
    model = MOTorch(
        module_type=    LinModel,
        in_shape=       256,
        out_shape=      10,
        name_timestamp= True,
        seed=           121,
        in_drop=        0.1)
    name = model.name
    print(model.name)

    inp = np.random.random((5, 256)).astype(np.float32)
    out1 = model(inp)
    print(out1)
    model.save()

    loaded_model = MOTorch(name=name, seed=123)
    print(loaded_model.name)
    out2 = loaded_model(inp)
    print(out2)

    assert np.sum(out1['logits'].cpu().detach().numpy()) == np.sum(out2['logits'].cpu().detach().numpy())


def test_copy_saved():
    model = MOTorch(
        module_type=    LinModel,
        in_shape=       256,
        out_shape=      10,
        name_timestamp= True,
        seed=           121,
        in_drop=        0.1)
    name = model.name
    print(model)
    model.save()

    name_copied = f'{name}_copied'
    MOTorch.copy_saved(name_src=name, name_trg=name_copied)

    model = MOTorch(name=name_copied)
    print(model)


### ParaSave

def test_ParaSave_interface():
    model = MOTorch(
        module_type=    LinModel,
        loglevel=       20,
        in_shape=       12,
        out_shape=      12,
        in_drop=        0.0)

    point = model.get_point()
    print(f'model point: {point}')
    assert point['gxable'] == True and point['psdd'] == {}

    pms = model.get_managed_params()
    print(f'model.get_managed_params(): {pms}')

    orig_seed = model.seed
    print(f'orig_seed: {orig_seed}')
    model.save()

    MOTorch.oversave_point(name=model.name, seed=252)

    dna = model.load_point(name=model.name, save_topdir='other')
    print(dna)
    assert not dna

    dna = model.load_point(name=model.name)
    print(dna)
    for p in pms:
        if p not in dna: print(p)
        assert p in dna
    assert dna['seed'] == 252

    model = MOTorch(name='inne', module_type=LinModel, in_drop=0.0)
    print(f'not loaded model in_shape: {model.in_shape}')
    assert model.in_shape != 12

    model = MOTorch(module_type=LinModel, loglevel=10)
    print(model['in_shape'])
    assert model.in_shape == 12

    model = MOTorch(
        module_type=    LinModel,
        name_timestamp= True,
        family=         'c',
        in_shape=       12,
        out_shape=      12,
        in_drop=        0.0)
    model.save()
    print(model.name, model.family)

    model = MOTorch(name=model.name)
    assert model.in_shape == 12

    model.copy_saved_point(name_src=model.name, name_trg=f'{model.name}_copied')

    model.gx_saved_point(
        name_parentA=   model.name,
        name_parentB=   None,
        name_child=     'GXed')

    psdd = {'seed': [0, 1000]}
    model = MOTorch(module_type=LinModel, name='GXLin', psdd=psdd, in_drop=0.0)

    print(model.gxable)
    print(model.name)
    print(model.family)
    print(model.seed)
    dna = model.gx_point(parentA=model, prob_noise=0.0, prob_axis=0.0)
    print(dna['seed'])
    dna = model.gx_point(parentA=model, prob_noise=1.0, prob_axis=1.0)
    print(dna['seed'])


def test_params_resolution():
    model = MOTorch(module_type=LinModel, in_drop=0.1)
    print(model['seed'])
    print(model['in_shape'])
    assert model['seed'] == 121
    assert model['in_shape'] == 784

    model = MOTorch(module_type=LinModel, seed=151, in_drop=0.1)
    print(model['seed'])
    assert model['seed'] == 151

    model = MOTorch(module_type=LinModel, in_shape=24, out_shape=24, in_drop=0.1)
    print(model['seed'])
    assert model['seed'] == 121
    model.save()
    name = model.name

    model = MOTorch(name=name, seed=212)
    print(model['in_shape'])
    print(model['seed'])
    assert model['in_shape'] == 24
    assert model['seed'] == 212


def test_class_method():
    model = MOTorch(module_type=LinModel, in_drop=0.1)
    print(model.name)
    point_org = model.get_point()
    print(point_org)
    assert point_org['gc_first_avg']
    model.save()

    point = MOTorch.load_point(name=model.name)
    print(point)
    assert not point['gc_first_avg']

    point_org.pop('gc_first_avg')
    point.pop('gc_first_avg')
    assert point_org == point


### call

def test_base_creation_and_call():
    model = MOTorch(module_type=LinModel, in_drop=0.1)

    inp = np.random.random((5, 784)).astype(np.float32)
    lbl = np.random.randint(0, 9, 5)

    out = model(inp)
    logits = out['logits']
    assert logits.shape[0] == 5 and logits.shape[1] == 10

    out = model.loss(inp, lbl)
    loss = out['loss']
    metrics = model.metrics(**out)
    assert type(loss) is torch.Tensor
    assert type(metrics) is dict

    for _ in range(5):
        out = model.backward(inp, lbl)
        loss = out['loss']
        metrics = model.metrics(**out)
        print(model.train_step, loss, metrics)


def test_data_conv():
    model = MOTorch(module_type=LinModel, in_drop=0.1)

    for inp in [
        [0, 1, 2],
        [0.1, 0.2],
        [[0.1, 0.2], [0.1, 0.2]],
        np.random.rand(10),
        [np.random.rand(10), np.random.rand(10)],
    ]:
        out = model.convert(inp)
        print(type(inp), out.shape, out.dtype, out.device)


def test_optimizer():
    model = MOTorch(module_type=LinModel, in_drop=0.0)
    assert type(model.optimizer) == torch.optim.Adam

    model = MOTorch(module_type=LinModelOpt, in_drop=0.0)
    assert type(model.optimizer) == torch.optim.SGD
    print(model.optimizer)


def test_training_mode():
    model = MOTorch(module_type=LinModel, in_drop=0.8)
    assert not model.module.training

    inp = np.random.random((5, 784)).astype(np.float32)
    lbl = np.random.randint(0, 9, 5)

    logits_nt = model(inp)['logits']
    loss_nt = model.loss(inp, lbl)['loss']

    model.train(True)
    assert model.module.training
    assert not torch.equal(logits_nt, model(inp)['logits'])
    assert not torch.equal(loss_nt, model.loss(inp, lbl)['loss'])
    assert model.module.training

    model.train(False)
    assert not model.module.training
    assert torch.equal(logits_nt, model(inp)['logits'])
    assert torch.equal(loss_nt, model.loss(inp, lbl)['loss'])
    assert not model.module.training


def test_no_grad():
    model = MOTorch(module_type=LinModel, in_drop=0.1)

    inp = np.random.random((2, 784)).astype(np.float32)
    lbl = np.random.randint(0, 9, 2)

    logits = model(inp)['logits']
    print(logits.requires_grad)
    assert logits.requires_grad
    print(logits.grad_fn)
    assert logits.grad_fn is not None
    for name, param in model.module.named_parameters():
        print(f'param name:{name} shape:{param.shape} grad:{param.grad}')
    print()

    out = model.loss(inp, lbl)
    loss = out['loss']
    print(loss.requires_grad)
    assert loss.requires_grad
    print(loss.grad_fn)
    assert loss.grad_fn is not None
    for param in model.module.parameters():
        print(f'param shape: {param.shape}, grad: {param.grad}')
        assert param.grad is None
    print()

    loss.backward()
    for name, param in model.module.named_parameters():
        print(f'param name:{name} shape:{param.shape} grad.shape:{param.grad.shape}')
        assert param.grad is not None

    model.module.zero_grad()
    for name, param in model.module.named_parameters():
        assert param.grad is None


def test_train_step():
    model = MOTorch(name='modA', module_type=LinModel, in_drop=0.1)

    inp = np.random.random((5, 784)).astype(np.float32)
    lbl = np.random.randint(0, 9, 5)
    for _ in range(5):
        out = model.backward(inp, lbl)
    print(model.name, model.train_step)
    model.save()

    model = MOTorch(name=model.name)
    print(model.name, model.train_step)
    assert model.name == 'modA' and model.train_step == 5


def test_seed_of_torch():
    model = MOTorch(module_type=LinModel, seed=121, in_drop=0.1)
    inp = np.random.random((5, 784)).astype(np.float32)
    out1 = model(inp)
    print(model['seed'])
    print(out1)

    model = MOTorch(module_type=LinModel, seed=121, in_drop=0.1)
    out2 = model(inp)
    print(model['seed'])
    print(out2)

    assert np.sum(out1['logits'].cpu().detach().numpy()) == np.sum(out2['logits'].cpu().detach().numpy())


def test_hpmser_mode():
    model = MOTorch(module_type=LinModel, hpmser_mode=True, in_drop=0.1)
    with pytest.raises(MOTorchException):
        model.save()


# GX

def test_gx_ckpt():
    nameA = 'modA'
    nameB = 'modB'

    model = MOTorch(module_type=LinModel, name=nameA, seed=121, in_drop=0.1, device=None)
    model.save()
    model = MOTorch(module_type=LinModel, name=nameB, seed=121, in_drop=0.1, device=None)
    model.save()

    MOTorch.gx_ckpt(nameA=nameA, nameB=nameB, name_child=f'{nameA}_GXed')


def test_gx_saved():
    nameC = 'modC'
    nameD = 'modD'

    model = MOTorch(module_type=LinModel, name=nameC, family='a', seed=121, in_drop=0.1, device=None)
    model.save()
    model = MOTorch(module_type=LinModel, name=nameD, family='a', seed=121, in_drop=0.1, device=None)
    model.save()

    MOTorch.gx_saved(name_parentA=nameC, name_parentB=nameD, name_child=f'{nameC}_GXed')
