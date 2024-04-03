import numpy as np
from pypaq.lipytools.pylogger import get_pylogger, get_child
import torch
from typing import Tuple, Dict
import unittest

from torchness.motorch import MOTorch, Module, MOTorchException
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


class LinModelOpt(LinModel):

    def get_optimizer_def(self) -> Tuple[type(torch.optim.Optimizer), Dict]:
        return torch.optim.SGD, {'momentum': 0.666}


logger = get_pylogger(name='test_motorch', level=20)


class TestMOTorch(unittest.TestCase):

    def setUp(self) -> None:
        flush_tmp_dir()

    ### init / build

    def test_base_init(self):
        model = MOTorch(
            module_type=    LinModel,
            in_drop=        0.0,
            logger=         get_child(logger, change_level=-10),
        )
        print(model)
        self.assertTrue(model.size == 7850)
        self.assertFalse(model.module.training)
        self.assertTrue(type(model.module) is LinModel)
        self.assertTrue(model.dtype == torch.float32)

    def test_init_raises(self):
        self.assertRaises(Exception, MOTorch)
        kwargs = dict(name='LinModel')
        self.assertRaises(Exception, MOTorch, **kwargs)
        kwargs = dict(module_type=LinModel)
        self.assertRaises(Exception, MOTorch, **kwargs)

    def test_name_stamp(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=        0.1,
            logger=         logger)
        print(model['name'])
        self.assertTrue(model['name'] == 'LinModel_MOTorch')

        model = MOTorch(
            module_type=    LinModel,
            name=           'LinTest',
            in_drop=        0.1,
            logger=         logger)
        print(model['name'])
        self.assertTrue(model['name'] == 'LinTest')

        model = MOTorch(
            module_type=        LinModel,
            name_timestamp=     True,
            in_drop=            0.1,
            logger=             logger)
        print(model['name'])
        self.assertTrue(model['name'] != 'MOTorch_LinModel')
        self.assertTrue({d for d in '0123456789'} & set([l for l in model['name']]))

    def test_device(self):

        # force GPU
        model = MOTorch(
            module_type=    LinModel,
            device=         0,
            in_drop=        0.0,
            logger=         logger)
        dev = model.device
        print(dev)
        self.assertTrue(dev == 'cuda:0' if torch.cuda.is_available() else 'cpu')

        # GPU or CPU
        model = MOTorch(
            module_type=    LinModel,
            device=         -1,
            in_drop=        0.0,
            logger=         logger)
        dev = model.device
        print(dev)
        self.assertTrue('cuda' in dev or 'cpu' in dev)

    def test_logger(self):
        log = get_pylogger(
            name=       'test_motorch',
            level=      10,
            flat_child= True)
        model = MOTorch(
            module_type=    LinModel,
            device=         0,
            in_drop=        0.0,
            logger=         log)
        model.save()

    ### save / load / folder

    def test_save_load(self):
        model = MOTorch(
            module_type=    LinModel,
            loglevel=       10,
            in_drop=        0.1,
            logger=         logger)
        self.assertTrue(model['seed']==121 and model['baseLR']==0.0003)
        self.assertTrue('loss' not in model.get_managed_params())
        model.save()
        name = model.name

        print('\nsaved, now loading..')

        logger.setLevel(10)
        model = MOTorch(
            name=       name,
            loglevel=   10,
            logger=     logger)
        self.assertTrue(model['in_drop']==0.1 and model['in_shape']==784)
        logger.setLevel(10)

    def test_read_only(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=        0.1,
            logger=         logger)
        model.save()
        name = model.name

        model = MOTorch(
            name=       name,
            read_only=  True,
            logger=     logger)
        self.assertRaises(MOTorchException, model.save)

    def test_save_load_full(self):

        model = MOTorch(
            module_type=    LinModel,
            in_shape=       256,
            out_shape=      10,
            name_timestamp= True,
            seed=           121,
            in_drop=        0.1,
            logger=         logger)
        name = model.name
        print(model.name)

        inp = np.random.random((5, 256)).astype(np.float32)

        out1 = model(inp)
        print(out1)
        model.save()

        loaded_model = MOTorch(
            name=           name,
            seed=           123, # although different seed, model will load checkpoint
            logger=         logger)
        print(loaded_model.name)
        out2 = loaded_model(inp)
        print(out2)
        # print(loaded_model)

        self.assertTrue(np.sum(out1['logits'].cpu().detach().numpy()) == np.sum(out2['logits'].cpu().detach().numpy()))

    def test_copy_saved(self):

        model = MOTorch(
            module_type=    LinModel,
            in_shape=       256,
            out_shape=      10,
            name_timestamp= True,
            seed=           121,
            in_drop=        0.1,
            logger=         logger)
        name = model.name
        print(model)
        model.save()

        name_copied = f'{name}_copied'
        MOTorch.copy_saved(name_src=name, name_trg=name_copied)

        model = MOTorch(name=name_copied, logger=logger)
        print(model)

    ### ParaSave

    def test_ParaSave_interface(self):

        model = MOTorch(
            module_type=    LinModel,
            loglevel=       20,
            in_shape=       12,
            out_shape=      12,
            in_drop=        0.0,
            logger=         logger)

        point = model.get_point()
        print(f'model point: {point}')
        self.assertTrue(point['gxable'] == True and point['psdd'] == {})

        pms = model.get_managed_params()
        print(f'model.get_managed_params(): {pms}')

        orig_seed = model.seed
        print(f'orig_seed: {orig_seed}')
        model.save()

        MOTorch.oversave_point(
            name=   model.name,
            seed=   252)

        # this will not load
        dna = model.load_point(
            name=           model.name,
            save_topdir=    'other')
        print(dna)
        self.assertFalse(dna)

        # this will load
        dna = model.load_point(name=model.name)
        print(dna)
        for p in pms:
            if p not in dna: print(p)
            self.assertTrue(p in dna)
        self.assertTrue(dna["seed"]==252)

        # this model will not load
        model = MOTorch(
            name=           'inne',
            module_type=    LinModel,
            in_drop=        0.0,
            logger=         logger)
        print(f'not loaded model in_shape: {model.in_shape}')
        self.assertTrue(model.in_shape != 12)

        # this model will load from MOTORCH_DIR
        model = MOTorch(
            module_type=    LinModel,
            loglevel=       10,
            logger=         logger)
        print(model['in_shape'])
        self.assertTrue(model.in_shape == 12)

        model = MOTorch(
            module_type=    LinModel,
            name_timestamp= True,
            family=         'c',
            in_shape=       12,
            out_shape=      12,
            in_drop=        0.0,
            logger=         logger)
        model.save()
        print(model.name, model.family)

        model = MOTorch(name=model.name, logger=logger)
        self.assertTrue(model.in_shape == 12)

        model.copy_saved_point(name_src=model.name, name_trg=f'{model.name}_copied')

        model.gx_saved_point(
            name_parentA=   model.name,
            name_parentB=   None,
            name_child=     'GXed')

        psdd = {'seed': [0,1000]}
        model = MOTorch(
            module_type=    LinModel,
            name=           'GXLin',
            psdd=           psdd,
            in_drop=        0.0,
            logger=         logger)

        print(model.gxable)
        print(model.name)
        print(model.family)
        print(model.seed)
        dna = model.gx_point(
            parentA=    model,
            prob_noise= 0.0,
            prob_axis=  0.0)
        print(dna['seed'])
        dna = model.gx_point(
            parentA=    model,
            prob_noise= 1.0,
            prob_axis=  1.0)
        print(dna['seed'])

    def test_params_resolution(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=        0.1,
            logger=         logger)
        print(model['seed'])        # value from MOTORCH_DEFAULTS
        print(model['in_shape'])    # value from module_type defaults
        self.assertTrue(model['seed'] == 121)
        self.assertTrue(model['in_shape'] == 784)

        model = MOTorch(
            module_type=    LinModel,
            seed=           151,
            in_drop=        0.1,
            logger=         logger)
        print(model['seed'])        # MOTORCH_DEFAULTS overridden with kwargs
        self.assertTrue(model['seed'] == 151)

        model = MOTorch(
            module_type=    LinModel,
            in_shape=       24,
            out_shape=      24,
            in_drop=        0.1,
            logger=         logger)
        print(model['seed'])        # MOTORCH_DEFAULTS overridden with module_type defaults
        self.assertTrue(model['seed'] == 121)
        model.save()
        name = model.name

        model = MOTorch(
            name=   name,
            seed=   212,
            logger= logger)
        print(model['in_shape'])    # loaded from save
        print(model['seed'])        # saved overridden with kwargs
        self.assertTrue(model['in_shape'] == 24)
        self.assertTrue(model['seed'] == 212)

    def test_class_method(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=        0.1,
            logger=         logger)
        print(model.name)
        point_org = model.get_point()
        print(point_org)
        self.assertTrue(point_org['gc_first_avg'])
        model.save()

        point = MOTorch.load_point(name=model.name)
        print(point)
        self.assertTrue(not point['gc_first_avg']) # INFO: gc_first_avg is updated while MOTorch.save()

        point_org.pop('gc_first_avg')
        point.pop('gc_first_avg')
        self.assertTrue(point_org == point)

    ### call

    def test_base_creation_and_call(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=        0.1,
            logger=         logger)

        inp = np.random.random((5,784)).astype(np.float32)
        lbl = np.random.randint(0,9,5)

        out = model(inp)
        print(out)
        logits = out['logits']
        self.assertTrue(logits.shape[0]==5 and logits.shape[1]==10)

        out = model.loss(inp, lbl)
        loss = out['loss']
        acc = out['acc']
        print(loss, acc)
        self.assertTrue(type(loss) is torch.Tensor)
        self.assertTrue(type(acc) is torch.Tensor)

        for _ in range(5):
            out = model.backward(inp, lbl)
            loss = out['loss']
            acc = out['acc']
            print(model.train_step, loss, acc)

    def test_data_conv(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=        0.1,
            logger=         logger)

        for inp in [
            [0, 1, 2],
            [0.1, 0.2],
            [[0.1,0.2],[0.1,0.2]],
            np.random.rand(10),
            [np.random.rand(10),np.random.rand(10)]
        ]:
            out = model.convert(inp)
            print(type(inp), out.shape, out.dtype, out.device)

    def test_optimizer(self):

        _log = get_child(logger, change_level=-10)

        model = MOTorch(
            module_type=    LinModel,
            in_drop=        0.0,
            logger=         _log,
        )
        self.assertTrue(type(model._opt) == torch.optim.Adam)

        model = MOTorch(
            module_type=    LinModelOpt,
            in_drop=        0.0,
            logger=         _log,
        )
        self.assertTrue(type(model._opt) == torch.optim.SGD)
        print(model._opt)

    def test_training_mode(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=        0.8,
            logger=         logger)
        # INFO: this model has dropout, so output should differ fro training True / False

        self.assertFalse(model.module.training) # default value

        inp = np.random.random((5,784)).astype(np.float32)
        lbl = np.random.randint(0,9,5)

        logits_nt = model(inp)['logits']
        loss_nt = model.loss(inp, lbl)['loss']

        model.train(True)
        self.assertTrue(model.module.training)
        self.assertFalse(torch.equal(logits_nt, model(inp)['logits']))
        self.assertFalse(torch.equal(loss_nt, model.loss(inp, lbl)['loss']))
        self.assertTrue(model.module.training)

        model.train(False)
        self.assertFalse(model.module.training)
        self.assertTrue(torch.equal(logits_nt, model(inp)['logits']))
        self.assertTrue(torch.equal(loss_nt, model.loss(inp, lbl)['loss']))
        self.assertFalse(model.module.training)

        self.assertFalse(torch.equal(logits_nt, model(inp, set_training=True)['logits']))
        self.assertFalse(torch.equal(loss_nt, model.loss(inp, lbl, set_training=True)['loss']))
        self.assertFalse(model.module.training)

    def test_no_grad(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=        0.1,
            logger=         logger)

        inp = np.random.random((2, 784)).astype(np.float32)
        lbl = np.random.randint(0, 9, 2)

        logits = model(inp)['logits']
        print(logits.requires_grad)
        self.assertTrue(not logits.requires_grad)
        print(logits.grad_fn)
        self.assertTrue(logits.grad_fn is None)
        for param in model.module.parameters():
            print(f'param shape: {param.shape}, grad: {param.grad}')
            self.assertTrue(param.grad is None)
        print()

        logits = model(inp, no_grad=False)['logits']
        print(logits.requires_grad)
        self.assertTrue(logits.requires_grad)
        print(logits.grad_fn)
        print(type(logits.grad_fn))
        self.assertTrue(logits.grad_fn is not None)
        for param in model.module.parameters():
            print(f'param shape: {param.shape}, grad: {param.grad}')
            self.assertTrue(param.grad is None)
        print()

        out = model.loss(inp, lbl)
        loss = out['loss']
        print(loss.requires_grad)
        self.assertTrue(loss.requires_grad)
        print(loss.grad_fn)
        self.assertTrue(loss.grad_fn is not None)
        for param in model.module.parameters():
            print(f'param shape: {param.shape}, grad: {param.grad}')
            self.assertTrue(param.grad is None)
        print()

        loss.backward()
        for param in model.module.parameters():
            print(f'param shape: {param.shape}, grad: {param.grad}')
            self.assertTrue(param.grad is not None)

        model.module.zero_grad()

        out = model.loss(inp, lbl, no_grad=True)
        self.assertRaises(RuntimeError, loss.backward)

    def test_train_step(self):

        model = MOTorch(
            name=           'modA',
            module_type=    LinModel,
            in_drop=        0.1,
            logger=         logger)

        inp = np.random.random((5, 784)).astype(np.float32)
        lbl = np.random.randint(0, 9, 5)
        for _ in range(5):
            out = model.backward(inp, lbl)
        print(model.name, model.train_step)
        model.save()

        model = MOTorch(name=model.name, logger=logger)
        print(model.name, model.train_step)
        self.assertTrue(model.name == 'modA' and model.train_step == 5)

    def test_seed_of_torch(self):

        model = MOTorch(
            module_type=    LinModel,
            seed=       121,
            in_drop=    0.1,
            logger=     logger)

        inp = np.random.random((5,784)).astype(np.float32)
        out1 = model(inp)
        print(model['seed'])
        print(out1)

        model = MOTorch(
            module_type=    LinModel,
            seed=           121,
            in_drop=        0.1,
            logger=         logger)

        out2 = model(inp)
        print(model['seed'])
        print(out2)

        self.assertTrue(np.sum(out1['logits'].cpu().detach().numpy()) == np.sum(out2['logits'].cpu().detach().numpy()))

    def test_hpmser_mode(self):

        model = MOTorch(
            module_type=    LinModel,
            hpmser_mode=    True,
            in_drop=        0.1,
            logger=         logger)
        self.assertRaises(MOTorchException, model.save)

    # GX

    def test_gx_ckpt(self):

        nameA = 'modA'
        nameB = 'modB'

        model = MOTorch(
            module_type=    LinModel,
            name=           nameA,
            seed=           121,
            in_drop=        0.1,
            device=         None,
            logger=         logger)
        model.save()

        model = MOTorch(
            module_type=    LinModel,
            name=           nameB,
            seed=           121,
            in_drop=        0.1,
            device=         None,
            logger=         logger)
        model.save()

        MOTorch.gx_ckpt(
            nameA=          nameA,
            nameB=          nameB,
            name_child=     f'{nameA}_GXed')

    def test_gx_saved(self):

        nameC = 'modC'
        nameD = 'modD'

        model = MOTorch(
            module_type=    LinModel,
            name=           nameC,
            family=         'a',
            seed=           121,
            in_drop=        0.1,
            device=         None,
            logger=         logger)
        model.save()

        model = MOTorch(
            module_type=    LinModel,
            name=           nameD,
            family=         'a',
            seed=           121,
            in_drop=        0.1,
            device=         None,
            logger=         logger)
        model.save()

        MOTorch.gx_saved(
            name_parentA=   nameC,
            name_parentB=   nameD,
            name_child=     f'{nameC}_GXed')