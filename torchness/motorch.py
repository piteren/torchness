import numpy as np
import shutil
import torch
from typing import Optional, Dict, Tuple, Any, Union, List

from pypaq.lipytools.printout import stamp, ProgBar
from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.moving_average import MovAvg
from pypaq.pms.base import get_class_init_params, point_trim
from pypaq.pms.parasave import ParaSave
from torchness.batcher import DataBatcher
from torchness.base import TNS, DTNS, NPL
from torchness.tools import accuracy, f1
from torchness.devices import get_devices
from torchness.ckpt import mrg_ckpts
from torchness.scaled_LR import ScaledLR
from torchness.grad_clipping import GradClipperMAVG
from torchness.tbwr import TBwr


class MOTorchException(Exception):
    pass


class Module(torch.nn.Module):
    """NN Module class supported by MOTorch

    default_score: name of metric used to compare modules
    default_score_format: formatting of default_score for nice print
    score_should_increase: direction of score improvement, e.g for loss False"""
    default_score = 'f1'
    default_score_format = '.5f'
    score_should_increase = True

    def __init__(self, logger=None, loglevel=20):
        super().__init__()
        if not logger:
            logger = get_pylogger(name=f'{self.__class__.__name__}_logger', level=loglevel)
        self.logger = logger

    def get_optimizer_definition(self) -> Tuple[type(torch.optim.Optimizer), Dict]:
        """if implemented, MOTorch will use Optimizer definition returned:
        Tuple[optimizer type, optimizer kwargs]

        * optimizer class may be given with kwarg (opt_class) to MOTorch,
        but if it is needed to define optimizer with its kwargs, this is the way"""
        raise MOTorchException(f'get_optimizer_definition not implemented for {self.__class__.__name__}')

    def forward(self, **kwargs) -> DTNS:
        """forward (FWD) pass
        returned DTNS should have at least 'logits' key

        exemplary implementation:
        return {'logits': self.logits(**kwargs)}"""
        raise MOTorchException(f'forward not implemented for {self.__class__.__name__}')

    def loss(self, **kwargs) -> DTNS:
        """forward (FWD) pass + loss
        returned DTNS should be: .forward() DTNS updated with 'loss'

        exemplary implementation:
        out = self(**kwargs)
        out['true'] = kwargs['true']
        out['loss'] = torch.nn.functional.cross_entropy(out['logits'], out['true'], reduction='mean')
        return out"""
        raise MOTorchException(f'loss not implemented for {self.__class__.__name__}')

    # noinspection PyMethodMayBeStatic
    def metrics(self, **kwargs) -> DTNS:
        """module metrics computation based on self.loss() output
        default implementation for [loss, acc, f1] for a classification model"""
        pred = torch.argmax(kwargs['logits'], dim=-1)
        return {
            'loss': kwargs['loss'],
            'accuracy': accuracy(target=kwargs['true'], pred=pred, logits=None),
            'f1': f1(target=kwargs['true'], pred=pred, logits=None, average='weighted')}


class MOTorch(ParaSave):
    """MOTorch holds Neural Network (NN) computation graph defined by Module

    - builds given graph defined by Module
    - manages MOTorch folder (subfolder of SAVE_TOPDIR named with MOTorch name)
      for all MOTorch data (logs, params, checkpoints), MOTorch supports
      serialization into / from this folder
    - extends ParaSave, manages all init parameters, properly resolves parameters
      using all possible sources:
        - defaults of MOTorch
        - defaults of Module
        - values saved in folder
        - values given by user to MOTorch init
    - parameters are kept in self as a Subscriptable to be easily accessed
    - properly resolves and holds name of object, adds stamp if needed
    - implements logger
    - may be read only (prevents save over)

    - manages:
        - devices: GPU / CPU with device: DevicesTorchness parameter
        - seed -> guarantees reproducibility
        - data format / type preparation (to be compatible with Module)
    - implements forward (FWD, with __call__) and loss call
    - implements backward (BWD) call -> runs gradient computation, clipping and backprop with given data

    - implements baseline training & testing with data loaded to Batcher
    - adds TensorBoard logging
    - supports hpmser mode
    - implements GX (genetic crossing)
    - adds some sanity checks

    MOTorch defaults are stored in MOTORCH_DEFAULTS dict and cannot be placed in __init__ defaults.
    This is a consequence of the params resolution mechanism in MOTorch / ParaSave,
    where parameters may come from four sources, and each subsequent source overrides the previous ones:
        1. __init__ defaults - only a few of them are considered in ParaSave managed params
        2. Module __init__ defaults
        3. saved in the folder
        4. provided through kwargs in __init__
    If all MOTorch parameters were set with __init__ defaults,
    it would not be possible to distinguish between sources 1 and 4.

    @DynamicAttrs <-- disables warning for unresolved attributes references"""

    MOTORCH_DEFAULTS = {
        'seed':             123,                # seed for torch and numpy
        'device':           -1,                 # :DevicesTorchness (check torchness.devices)
        'dtype':            torch.float32,
        'bypass_data_conv': False,              # to bypass input data conversion with when calling: __call__, loss, backward
            # training
        'batch_size':       64,                 # training batch size
        'n_batches':        1000,               # default length of training
        'opt_class':        torch.optim.Adam,   # default optimizer
        'train_step':       0,                  # default (starting) train step, updated with backward()
            # LR management (check torchness.base_elements.ScaledLR)
        'baseLR':           3e-4,
        'warmup_end':       0,
        'anneal_start':     None,
        'anneal_base':      0.999,
        'anneal_mul':       1.0,
            # gradients clipping parameters (check torchness.grad_clipping.GradClipperMAVG)
        'gc_do_clip':       False,
        'gc_start_val':     0.1,
        'gc_factor':        0.01,
        'gc_first_avg':     True,
        'gc_max_clip':      None,
        'gc_max_upd':       1.5,
            # other
        'try_load_ckpt':    True,               # tries to load a checkpoint while init
        'hpmser_mode':      False,              # it will set model to be read_only and quiet when running with hpmser
        'read_only':        False,              # sets MOTorch to be read only - won't save anything (won't even create self.motorch_dir)
        'do_TB':            True,               # runs TensorBard, saves in self.motorch_dir
    }

    # override ParaSave defaults
    SAVE_TOPDIR = '_models'         # save top directory
    SAVE_FN_PFX = 'motorch_point'   # POINT file prefix

    def __init__(
            self,
            module_type: Optional[type(Module)]=    None,
            name: Optional[str]=                    None,
            name_timestamp=                         False,
            save_topdir: Optional[str]=             None,
            save_fn_pfx: Optional[str]=             None,
            tbwr: Optional[TBwr]=                   None,
            logger=                                 None,
            loglevel=                               20,
            flat_child=                             False,
            **kwargs):

        # TODO: temporary, delete later
        if 'devices' in kwargs:
            raise MOTorchException('\'devices\' param is not supported by MOTorch, please use \'device\'')

        if not (name or module_type):
            raise MOTorchException('name OR module_type must be given!')

        name = self._get_name(
            module_type=    module_type,
            name=           name,
            name_timestamp= name_timestamp)

        if not save_topdir: save_topdir = self.SAVE_TOPDIR
        if not save_fn_pfx: save_fn_pfx = self.SAVE_FN_PFX

        # some early kwargs overrides

        if kwargs.get('hpmser_mode', False):
            loglevel = 50
            kwargs['read_only'] = True

        if kwargs.get('read_only', False):
            kwargs['do_TB'] = False

        _read_only = kwargs.get('read_only', False)

        self.logger = logger or get_pylogger(
            name=       name,
            add_stamp=  False,
            folder=     None if _read_only else self._get_model_dir(model_name=name, save_topdir=save_topdir),
            level=      loglevel,
            flat_child= flat_child)

        self.logger.info(f'*** MOTorch : {name} *** initializes ..')
        self.logger.info(f'> {name} save_topdir: {save_topdir}{" <- read only mode!" if _read_only else ""}')

        # init as a ParaSave
        super().__init__(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx,
            logger=         get_child(self.logger, 'ParaSave_logger'),
            **kwargs)
        point_saved = self.get_point()

        # **************************************************************************************** further resolve POINT

        ### resolve module_type

        module_type_saved = point_saved.get('module_type', None)

        if not module_type and not module_type_saved:
            msg = 'module_type was not given and has not been found in saved, cannot continue!'
            self.logger.error(msg)
            raise MOTorchException(msg)

        if module_type and module_type_saved and module_type != module_type_saved:
            self.logger.info('given module_type differs from module_type found in saved, using saved')

        module_type = module_type_saved or module_type
        self.logger.info(f'> {self.name} module_type: {module_type.__name__}')

        _module_init_def = get_class_init_params(module_type)['with_defaults'] # defaults of self.module_type.__init__

        ### update in proper order

        self._point = {}
        self._point.update(ParaSave.PARASAVE_DEFAULTS)
        self._point.update(self.MOTORCH_DEFAULTS)
        self._point.update(_module_init_def)
        self._point.update(point_saved)
        self._point.update(kwargs)
        self._point["module_type"] = module_type

        # remove logger (may come from Module init defaults)
        if 'logger' in self._point:
            self._point.pop('logger')

        ### finally resolve device

        # device parameter, may be given to MOTorch in DevicesTorchness type
        # it is cast to PyTorch namespace here
        self.logger.debug(f'> {self.name} resolves devices, given: {self._point["device"]}')
        self.logger.debug(f'> torch.cuda.is_available(): {torch.cuda.is_available()}')
        devices = get_devices(
            devices=            self._point["device"],
            torch_namespace=    True,
            logger=             get_child(self.logger, 'get_devices'))
        if not devices:
            self.logger.warning(f'given device: {self._point["device"]} is not available, using CPU')
            devices = ['cpu']
        device = devices[0]
        self.logger.info(f'> {self.name} given devices: {self._point["device"]}, will use: {device}')
        self._point['device'] = device

        ### prepare Module point and extract not used kwargs

        self._module_point = point_trim(module_type, self._point)
        self._module_point['logger'] = get_child(self.logger, 'Moduleloggerger')

        rep = (f'{self.name} POINT sources:\n'
               f'> PARASAVE_DEFAULTS:        {ParaSave.PARASAVE_DEFAULTS}\n'
               f'> MOTORCH_DEFAULTS:         {self.MOTORCH_DEFAULTS}\n'
               f'> Module.__init__ defaults: {_module_init_def}\n'
               f'> POINT saved:              {point_saved}\n'
               f'> given kwargs:             {kwargs}\n'
               f'Module complete POINT:      {self._module_point}\n'
               f'MOTorch complete POINT:     {self._point}')
        self.logger.debug(rep)

        _kwargs_not_used = {}
        out = get_class_init_params(MOTorch)
        motorch_init_params = out['without_defaults'] + list(out['with_defaults'].keys())
        motorch_params_all = list(ParaSave.PARASAVE_DEFAULTS.keys()) + list(self.MOTORCH_DEFAULTS.keys()) + motorch_init_params
        for k in kwargs:
            if k not in self._module_point and k not in motorch_params_all:
                _kwargs_not_used[k] = kwargs[k]
        if _kwargs_not_used:
            self.logger.warning(f'> there are kwargs given but not used by MOTorch nor Module: {_kwargs_not_used}')

        self.update(self._point)

        # parameters names safety check
        found = self.check_params_sim(params=list(self.MOTORCH_DEFAULTS.keys()) + list(kwargs.keys()))
        if found:
            self.logger.warning(f'{self.name} (MOTorch) was asked to check for params similarity and found:')
            for pa, pb in found:
                self.logger.warning(f'> params \'{pa}\' and \'{pb}\' are close !!!')

        # set seed in all possible areas (https://pytorch.org/docs/stable/notes/randomness.html)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        ### build MOTorch Module

        self.logger.info(f'{self.name} builds graph of {self.module_type.__name__}')
        self.module = self.module_type(**self._module_point)

        if self.try_load_ckpt:
            self.load_ckpt()
        else:
            self.logger.info(f'> {self.name} checkpoint not loaded, not even tried because \'try_load_ckpt\' was set to {self.try_load_ckpt}')

        self.logger.debug(f'> moving {self.name} to device: {self.device}, dtype: {self.dtype}')
        self.module.to(self.device)
        self.module.to(self.dtype)

        self.logger.debug(f'{self.name} Module initialized!')

        ### resolve optimizer

        opt_kwargs = {}
        try:
            self.opt_class, opt_kwargs = self.module.get_optimizer_definition()
            self.logger.debug(f'using optimizer from Module: {self.opt_class.__name__}, Module optimizer kwargs: {opt_kwargs}')
        except MOTorchException:
            self.logger.debug(f'using optimizer resolved by MOTorch: {self.opt_class.__name__}')

        opt_kwargs['params'] = self.module.parameters()
        opt_kwargs['lr'] = self.baseLR
        self._opt = self.opt_class(**opt_kwargs)
        self.logger.debug(f'MOTorch optimizer:\n{self._opt}')

        # from now LR is managed by scheduler
        self._scheduler = ScaledLR(
            optimizer=      self._opt,
            step=           self.train_step,
            warmup_end=     self.warmup_end,
            anneal_start=   self.anneal_start,
            anneal_base=    self.anneal_base,
            anneal_mul=     self.anneal_mul,
            logger=         get_child(self.logger, 'ScaledLR'))

        self._grad_clipper = GradClipperMAVG(
            do_clip=        self.gc_do_clip,
            module=         self.module,
            start_val=      self.gc_start_val,
            factor=         self.gc_factor,
            first_avg=      self.gc_first_avg,
            max_clip=       self.gc_max_clip,
            max_upd=        self.gc_max_upd,
            logger=         get_child(self.logger, 'GradClipperMAVG'))

        # MOTorch by default is not in training mode
        self.train(False)
        self.logger.debug(f'> set {self.name} train.mode to False ..')

        # TensorBoard writer
        self._TBwr = tbwr or TBwr(
            logdir=self._get_model_dir(
                model_name=     self.name,
                save_topdir=    self.save_topdir)) if self.do_TB else None

        self._batcher = None

        self.logger.debug(str(self))
        self.logger.info(f'MOTorch init finished!')

    def exclude_from_params(self) -> List[str]:
        return super().exclude_from_params() + ['module']

    @classmethod
    def _get_name(
            cls,
            module_type: Optional[type(Module)]=    None,
            name: Optional[str]=                    None,
            name_timestamp=                         False,
    ) -> str:
        """resolves MOTorch name"""
        # resolve name
        if not name:
            name = f'{module_type.__name__}_{cls.__name__}'
        if name_timestamp:
            name += f'_{stamp()}'
        return name

    # **************************************************************************** model call (run NN with data) methods

    def convert(self, data:Any) -> TNS:
        """converts given data to TNS compatible with self (device,dtype)"""

        # do not convert None
        if type(data) is not None:

            if type(data) is not torch.Tensor:
                if type(data) is np.ndarray: data = torch.from_numpy(data)
                else:                        data = torch.tensor(data)

            # convert device + float types
            data = data.to(self.device, self.dtype if data.is_floating_point() or data.is_complex() else None)

        return data

    def __call__(self, *args, bypass_data_conv=False, **kwargs) -> DTNS:
        """forward (FWD) pass of self.module with data preparation"""
        if not (bypass_data_conv or self.bypass_data_conv):
            args = [self.convert(data=a) for a in args]
            kwargs = {k: self.convert(data=kwargs[k]) for k in kwargs}
        return self.module(*args, **kwargs)

    def loss(self, *args, bypass_data_conv=False, **kwargs) -> DTNS:
        """forward (FWD) pass + loss of self.module with data preparation"""
        if not (bypass_data_conv or self.bypass_data_conv):
            args = [self.convert(data=a) for a in args]
            kwargs = {k: self.convert(data=kwargs[k]) for k in kwargs}
        return self.module.loss(*args, **kwargs)

    def backward(self, *args, bypass_data_conv=False, **kwargs) -> DTNS:
        """backward call on NN, runs loss calculation + update parameters of NN with optimizer"""

        out = self.loss(*args, bypass_data_conv=bypass_data_conv, **kwargs)

        self._opt.zero_grad()               # clear gradients
        out['loss'].backward()              # build gradients
        gnD = self._grad_clipper.clip()     # clip gradients, adds: 'gg_norm' & 'gg_norm_clip' to out
        self._opt.step()                    # apply optimizer
        self._scheduler.step()              # apply LR scheduler
        self.train_step += 1                # update step

        out['currentLR'] = self._scheduler.get_lr()[0] # INFO: currentLR of the first group is taken
        out.update(gnD)

        return out

    # *********************************************************************************************** load / save / copy

    @classmethod
    def _get_model_dir(cls, model_name:str, save_topdir:Optional[str]=None) -> str:
        """returns model directory path"""
        if not save_topdir: save_topdir = cls.SAVE_TOPDIR
        return f'{save_topdir}/{model_name}'

    @classmethod
    def _get_ckpt_path(cls, model_name:str, save_topdir:Optional[str]=None) -> str:
        """returns path of checkpoint pickle file"""
        model_dir = cls._get_model_dir(model_name=model_name, save_topdir=save_topdir)
        return f'{model_dir}/{model_name}.pt'

    def load_ckpt(
            self,
            name: Optional[str]=        None,  # allows to load custom name (model_name)
            save_topdir: Optional[str]= None,  # allows to load from custom save_topdir
    ) -> Optional[dict]:
        """tries to load checkpoint and return additional data"""

        ckpt_path = self._get_ckpt_path(
            model_name=     name or self.name,
            save_topdir=    save_topdir or self.save_topdir)

        save_obj = None

        try:
            save_obj = torch.load(f=ckpt_path, map_location=self.device, weights_only=True) # immediately place all tensors to current device (not previously saved one)
            self.module.load_state_dict(save_obj.pop('model_state_dict'))
            self.logger.info(f'> {self.name} checkpoint loaded from {ckpt_path}')
        except Exception as e:
            # this exception logs as INFO since it is quite normal to not load checkpoint while init
            self.logger.info(f'> {self.name} checkpoint NOT loaded because of exception: {e}')

        return save_obj

    def save_ckpt(
            self,
            name: Optional[str]=                None,   # allows to save under custom name (model_name)
            save_topdir: Optional[str]=         None,   # allows to save in custom save_topdir
            additional_data: Optional[Dict]=    None,   # allows to save additional
    ) -> None:
        """saves model checkpoint & optionally additional data"""

        ckpt_path = self._get_ckpt_path(
            model_name=     name or self.name,
            save_topdir=    save_topdir or self.save_topdir)

        save_obj = {'model_state_dict': self.module.state_dict()}
        if additional_data: save_obj.update(additional_data)

        torch.save(obj=save_obj, f=ckpt_path)

    def save(self):
        """saves MOTorch (ParaSave POINT and model checkpoint)"""

        if self.read_only:
            raise MOTorchException('read_only MOTorch cannot be saved!')

        # to properly start grad clipping after load
        self['gc_first_avg'] = False
        self['gc_start_val'] = float(self._grad_clipper.mavg())

        self.save_point()
        self.save_ckpt()
        self.logger.info(f'{self.__class__.__name__} {self.name} saved to {self.save_topdir}')

    @classmethod
    def copy_checkpoint(
            cls,
            name_src: str,
            name_trg: str,
            save_topdir_src: Optional[str]= None,
            save_topdir_trg: Optional[str]= None):
        if not save_topdir_src: save_topdir_src = cls.SAVE_TOPDIR
        if not save_topdir_trg: save_topdir_trg = save_topdir_src
        shutil.copyfile(
            src=cls._get_ckpt_path(model_name=name_src, save_topdir=save_topdir_src),
            dst=cls._get_ckpt_path(model_name=name_trg, save_topdir=save_topdir_trg))

    @classmethod
    def copy_saved(
            cls,
            name_src: str,
            name_trg: str,
            save_topdir_src: Optional[str]= None,
            save_topdir_trg: Optional[str]= None,
            save_fn_pfx: Optional[str]=     None,
            device=                         None,
            logger=                         None,
            loglevel=                       30):
        """copies full MOTorch folder (POINT & checkpoints)"""

        if not save_topdir_src: save_topdir_src = cls.SAVE_TOPDIR
        if save_topdir_trg is None: save_topdir_trg = save_topdir_src
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX

        cls.copy_saved_point(
            name_src=           name_src,
            name_trg=           name_trg,
            save_topdir_src=    save_topdir_src,
            save_topdir_trg=    save_topdir_trg,
            save_fn_pfx=        save_fn_pfx,
            logger=             logger,
            loglevel=           loglevel,
            device=             device)

        cls.copy_checkpoint(
            name_src=           name_src,
            name_trg=           name_trg,
            save_topdir_src=    save_topdir_src,
            save_topdir_trg=    save_topdir_trg)

    # *************************************************************************************************************** GX

    @classmethod
    def gx_ckpt(
            cls,
            nameA: str,                     # name parent A
            nameB: str,                     # name parent B
            name_child: str,                # name child
            save_topdirA: Optional[str]=        None,
            save_topdirB: Optional[str]=        None,
            save_topdir_child: Optional[str]=   None,
            ratio: float=                       0.5,
            noise: float=                       0.03,
    ):
        """GX on 2 checkpoints only of saved 2 MOTorch"""

        if not save_topdirA: save_topdirA = cls.SAVE_TOPDIR
        if not save_topdirB: save_topdirB = save_topdirA
        if not save_topdir_child: save_topdir_child = save_topdirA

        prep_folder(f'{save_topdir_child}/{name_child}')

        mrg_ckpts(
            ckptA=cls._get_ckpt_path(model_name=nameA,      save_topdir=save_topdirA),
            ckptB=cls._get_ckpt_path(model_name=nameB,      save_topdir=save_topdirB),
            ckptM=cls._get_ckpt_path(model_name=name_child, save_topdir=save_topdir_child),
            ratio=ratio,
            noise=noise)

    @classmethod
    def gx_saved(
            cls,
            name_parentA: str,
            name_parentB: Optional[str],    # if not given makes GX only with main parent
            name_child: str,
            save_topdir_parentA: Optional[str]= None,
            save_topdir_parentB: Optional[str]= None,
            save_topdir_child: Optional[str]=   None,
            save_fn_pfx: Optional[str]=         None,
            device=                             None,
            do_gx_ckpt=                         True,
            ratio: float=                       0.5,
            noise: float=                       0.03,
            logger=                             None,
            loglevel=                           30,
    ) -> None:
        """performs GX on saved MOTorch"""

        if not save_topdir_parentA: save_topdir_parentA = cls.SAVE_TOPDIR
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX

        cls.gx_saved_point(
            name_parentA=           name_parentA,
            name_parentB=           name_parentB,
            name_child=             name_child,
            save_topdir_parentA=    save_topdir_parentA,
            save_topdir_parentB=    save_topdir_parentB,
            save_topdir_child=      save_topdir_child,
            save_fn_pfx=            save_fn_pfx,
            logger=                 logger,
            loglevel=               loglevel)

        if do_gx_ckpt:
            cls.gx_ckpt(
                nameA=              name_parentA,
                nameB=              name_parentB or name_parentA,
                name_child=         name_child,
                save_topdirA=       save_topdir_parentA,
                save_topdirB=       save_topdir_parentB,
                save_topdir_child=  save_topdir_child,
                ratio=              ratio,
                noise=              noise)
        # build and save to have checkpoint saved
        else:
            child = cls(
                name=               name_child,
                save_topdir=        save_topdir_child or save_topdir_parentA,
                save_fn_pfx=        save_fn_pfx,
                device=             device,
                logger=             logger,
                loglevel=           loglevel)
            child.save()

    # ******************************************************************** train / test, exposed module methods to self

    def load_data(
            self,
            data_TR: Dict[str,np.ndarray],
            data_TS: Optional[Union[Dict[str,NPL], Dict[str,Dict[str,NPL]]]]=   None,
            split_factor: float=                                                0.0):
        """converts and loads data to Batcher"""

        data_TR = {k: self.convert(data_TR[k]) for k in data_TR}

        if data_TS:
            # named test-set
            if type(list(data_TS.values())[0]) is dict:
                for k in data_TS:
                    data_TS[k] = {sk: self.convert(data_TS[k][sk]) for sk in data_TS[k]}
            else:
                data_TS = {k: self.convert(data_TS[k]) for k in data_TS}

        self._batcher = DataBatcher(
            data_TR=        data_TR,
            data_TS=        data_TS,
            split_factor=   split_factor,
            batch_size=     self.batch_size,
            batching_type=  'random',
            seed=           self.seed,
            logger=         get_child(self.logger, 'Batcher'))

    def run_train(
            self,
            data_TR: Optional[Dict[str,np.ndarray]]=None,
            data_TS: Optional[Union[Dict[str,NPL], Dict[str,Dict[str,NPL]]]]=None,
            split_factor: float=        0.0,
            n_batches: Optional[int]=   None,
            test_freq=                  100,
            mov_avg_factor=             0.1,
            save_max=                   True,
            empty_cuda_cache: bool=     False,
        ) -> Optional[float]:
        """trains model, returns optional test score

        data_TR: accepts also Dict[str,torch.Tensor]
        test_freq: number of batches between tests
        save_max: saves model while training (after best test)
        empty_cuda_cache: empties cuda cache every batch
            may help reduce GPU memory usage, but
            may increase loop time"""

        if data_TR:
            self.load_data(data_TR=data_TR, data_TS=data_TS, split_factor=split_factor)

        if not self._batcher:
            raise MOTorchException(f'{self.name} has not been given data for training, use load_data()')

        if n_batches is None: n_batches = self.n_batches
        nfo = (f'{self.name} training starts:\n'
               f'> data sizes (TR,TS) samples: {self._batcher.get_data_size()}\n'
               f'> batch size: {self["batch_size"]}\n'
               f'> n_batches: {n_batches}')
        self.logger.info(nfo)

        tr_metrics_accumulated = {}

        ts_score_name = self.module.default_score
        ts_score_best = None                    # test score best value
        ts_score_all_results = []               # test score all results
        ts_score_mav = MovAvg(mov_avg_factor)   # test score moving average

        # initial save
        if not self.read_only and save_max:
            self.save_ckpt()

        ts_bIX = [bIX for bIX in range(n_batches+1) if not bIX % test_freq] # batch indexes when test will be performed
        if not ts_bIX:
            raise MOTorchException('model SHOULD BE tested while training, but no test indexes are given')
        ten_factor = int(0.1*len(ts_bIX)) # number of tests for last 10% of training
        if ten_factor < 1: ten_factor = 1 # we need at least one result
        if self.hpmser_mode: ts_bIX = ts_bIX[-ten_factor:]

        _ds = self.module.default_score
        _dsf = self.module.default_score_format

        _mode = self.training
        self.train()
        prog = ProgBar(n_batches, name=f"{self.name} train", logger=self.logger)
        batch_IX = 0
        while batch_IX < n_batches:

            out = self.backward(**self._batcher.get_batch(), bypass_data_conv=True)
            tr_metrics = self.metrics(**out)
            loss = out['loss']
            batch_IX += 1

            if self.do_TB:
                self.log_TB(value=loss,                tag='TR/loss',    step=self.train_step)
                self.log_TB(value=out['gg_norm'],      tag='TR/gn',      step=self.train_step)
                self.log_TB(value=out['gg_norm_clip'], tag='TR/gn_clip', step=self.train_step)
                self.log_TB(value=out['currentLR'],    tag='TR/cLR',     step=self.train_step)
                for k in tr_metrics:
                    if k != 'loss':
                        self.log_TB(value=tr_metrics[k], tag=f'TR/{k}', step=self.train_step)

            if not tr_metrics_accumulated:
                for k in tr_metrics:
                    tr_metrics_accumulated[k] = []

            for k in tr_metrics_accumulated:
                tr_metrics_accumulated[k].append(tr_metrics[k])

            if batch_IX in ts_bIX:

                res = self.run_test()
                first_key = list(res.keys())[0]

                for k in res:

                    ts_metrics = res[k]
                    ts_score = ts_metrics[_ds]
                    if ts_score_best is None:
                        ts_score_best = ts_score
                    ts_score_all_results.append(ts_score)

                    tr_metrics_accumulated = {k: sum(tr_metrics_accumulated[k]) / test_freq for k in tr_metrics_accumulated}
                    _ds_gain = ts_metrics[_ds]-tr_metrics_accumulated[_ds]

                    key_name = f'_{k}' if k != self._batcher.default_TS_name else ''
                    if self.do_TB:
                        for mk in ts_metrics:
                            self.log_TB(value=ts_metrics[mk], tag=f'TS{key_name}/{mk}', step=self.train_step)
                        self.log_TB(value=ts_score_mav.upd(ts_score), tag=f'TS{key_name}/{_ds}_mav', step=self.train_step)
                        self.log_TB(value=_ds_gain, tag=f'TS{key_name}/{_ds}_gain', step=self.train_step)

                    _tr_nfo = f'{_ds}:{tr_metrics_accumulated[_ds]:{_dsf}}'
                    _ts_nfo = f'{_ds}:{ts_metrics[_ds]:{_dsf}} {_ds}_gain:{_ds_gain:{_dsf}}'
                    prog_nfo = f'TR: {_tr_nfo} -- TS{key_name}: {_ts_nfo}'
                    prog(n=batch_IX, prefix=prog_nfo)
                    tr_metrics_accumulated = {}

                    # model is saved for best ts_score for the first_key (TS name)
                    if k==first_key and ts_score is not None and (
                            (ts_score > ts_score_best and self.module.score_should_increase) or
                            (ts_score < ts_score_best and not self.module.score_should_increase)):
                        ts_score_best = ts_score
                        if not self.read_only and save_max:
                            self.save_ckpt()

            if empty_cuda_cache:
                torch.cuda.empty_cache()

        self.train(_mode)

        ts_score_fin = None
        if save_max:
            ts_score_fin = ts_score_best
            self.logger.info(f"loading {self.name} checkpoint saved for max score ..")
            self.load_ckpt()

        # weighted (linear ascending weight) test score for last 10% test results
        else:
            if ts_score_all_results:
                ts_score_fin = 0.0
                weight = 1
                sum_weight = 0
                for tr in ts_score_all_results[-ten_factor:]:
                    ts_score_fin += tr*weight
                    sum_weight += weight
                    weight += 1
                ts_score_fin /= sum_weight

        if ts_score_fin is not None:
            self.logger.info(f'> test_{ts_score_name}_best: {ts_score_best}')
            self.logger.info(f'> test_{ts_score_name}_fin:  {ts_score_fin}')
            if self.do_TB:
                self.log_TB(value=ts_score_fin, tag=f'TS/ts_{ts_score_name}_fin', step=self.train_step)

        return ts_score_fin

    def run_test(
            self,
            data: Optional[Dict[str,np.ndarray]]=   None,
            split_factor: float=                    1.0,
    ) -> Dict[str,DTNS]:
        """tests model, returns dict {testset_name:DTNS} where DTNS are metrics"""

        _mode = self.training
        self.train(False)

        if data:
            self.load_data(data_TR=data, split_factor=split_factor)

        if not self._batcher:
            raise MOTorchException(f'{self.name} has not been given data for testing, use load_data() or give it while testing!')

        res = {}
        with torch.no_grad():
            for tn in self._batcher.get_TS_names():

                n_all = 0
                metrics_acc = {}
                for batch in self._batcher.get_TS_batches(name=tn):

                    out = self.loss(**batch, bypass_data_conv=True)

                    n_new = len(out['logits'])
                    n_all += n_new

                    metrics = self.metrics(**out)
                    for k in metrics:
                        if k not in metrics_acc:
                            metrics_acc[k] = []
                        metrics_acc[k].append(metrics[k]*n_new)

                for k in metrics_acc:
                    metrics_acc[k] = sum(torch.as_tensor(v) for v in metrics_acc[k]) / n_all

                res[tn] = metrics_acc

        self.train(_mode)

        return res

    def metrics(self, **kwargs) -> DTNS:
        return self.module.metrics(**kwargs)

    # *********************************************************************************************** other / properties

    def update_baseLR(self, lr: float):
        """updates scheduler baseLR of 0 group"""
        self.baseLR = lr
        self._scheduler.update_base_lr0(lr)

    def train(self, mode:bool=True):
        return self.module.train(mode)

    @property
    def training(self) -> bool:
        return self.module.training

    @property
    def tbwr(self):
        return self._TBwr

    def log_TB(self, value, tag:str, step:Optional[int]=None):
        """logs value to TB"""
        if step is None:
            step = self.train_step
        if self.do_TB:
            self._TBwr.add(value=value, tag=tag, step=step)
        else: self.logger.warning(f'{self.name} cannot log to TensorBoard since \'do_TB\' flag was set to False!')

    def log_histogram_TB(self, values, tag:str, step:Optional[int]=None, bins="tensorflow"):
        """logs values to TB histogram"""
        if step is None:
            step = self.train_step
        if self.do_TB:
            self._TBwr.add_histogram(values=values, tag=tag, step=step, bins=bins)
        else: self.logger.warning(f'{self.name} cannot log to TensorBoard since \'do_TB\' flag was set to False!')

    @property
    def optimizer(self):
        return self._opt

    @property
    def size(self) -> int:
        """returns number of module parameters"""
        return sum([p.numel() for p in self.module.parameters()])

    def __str__(self):
        s = f'{self.__class__.__name__} (MOTorch): {ParaSave.__str__(self)}\n'
        s += f'{str(self.module)}\n ### model size: {self.size} params'
        return s