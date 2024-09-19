import numpy as np
import shutil
from sklearn.metrics import f1_score
import torch
from typing import Optional, Dict, Tuple, Any, Union

from pypaq.lipytools.printout import stamp
from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.moving_average import MovAvg
from pypaq.pms.base import get_class_init_params, point_trim
from pypaq.pms.parasave import ParaSave
from torchness.comoneural.batcher import DataBatcher
from torchness.base import TNS, DTNS, NUM, NPL
from torchness.devices import get_devices
from torchness.ckpt import mrg_ckpts
from torchness.scaled_LR import ScaledLR
from torchness.grad_clipping import GradClipperMAVG
from torchness.tbwr import TBwr


class MOTorchException(Exception):
    pass


class Module(torch.nn.Module):
    """ Module type supported by MOTorch
    defines computation graph of forward (FWD) and loss
    accuracy() and f1() are metrics used by MOTorch while training """

    def __init__(self, logger=None, loglevel=20):
        super().__init__()
        if not logger:
            logger = get_pylogger(name=f'{self.__class__.__name__}_logger', level=loglevel)
        self.logger = logger


    def forward(self, **kwargs) -> DTNS:
        """ forward pass (FWD) function
        returned DTNS should have at least 'logits' key
        with logits tensor for proper MOTorch.run_train()

        exemplary implementation:
            return {'logits': self.logits(input)} """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def accuracy(self, logits:TNS, labels:TNS) -> NUM:
        """ baseline accuracy implementation for logits & lables """
        return (torch.argmax(logits, dim=-1) == labels).to(torch.float).mean()

    # noinspection PyMethodMayBeStatic
    def f1(self, logits:TNS, labels:TNS, average='weighted') -> float:
        """ baseline F1 implementation for logits & lables
        correctly supports average (since test while training may be run in batches):
            micro (per sample)
            macro (per class)
            weighted (per class weighted by support) """
        preds = torch.argmax(logits, dim=-1).cpu()
        return f1_score(
            y_true=         labels.cpu(),
            y_pred=         preds,
            average=        average,
            labels=         np.unique(preds),
            zero_division=  0)


    def get_optimizer_def(self) -> Tuple[type(torch.optim.Optimizer), Dict]:
        """ if implemented, MOTorch will use returned Optimizer definition
        (Optimizer type, Optimizer kwargs)

        *optimizer class may be given with kwarg (opt_class) to MOTorch,
        but if it is needed to define optimizer with its kwargs, this is the way """
        raise MOTorchException(f'get_optimizer_def not implemented for {self.__class__.__name__}')


    def loss(self, **kwargs) -> DTNS:
        """ forward + loss function
        returned DTNS should be: .forward() DTNS updated with loss (and optional acc, f1)

        exemplary implementation:
        out = self(input)                                                                   <- forward DTNS
        logits = out['logits']
        out['loss'] = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')   <- update with loss
        out['acc'] = self.accuracy(logits, labels)                                          <- update with acc
        out['f1'] = self.f1(logits, labels)                                                 <- update with f1 """
        raise NotImplementedError


class MOTorch(ParaSave):
    """ MOTorch holds Neural Network (NN) computation graph defined by Module

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
        - training mode (may be overridden by user)
        - data format / type preparation (to be compatible with Module)
        - automatic gradient computation switching while inference training
    - implements forward (FWD) call (with __call__)
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

    @DynamicAttrs <-- disables warning for unresolved attributes references """

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
        'warm_up':          None,
        'n_wup_off':        2.0,
        'ann_base':         None,
        'ann_step':         1.0,
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
            raise MOTorchException('\'devices\' param is no more supported by MOTorch, please use \'device\'')

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

        if not logger:
            logger = get_pylogger(
                name=       name,
                add_stamp=  False,
                folder=     None if _read_only else self._get_model_dir(model_name=name, save_topdir=save_topdir),
                level=      loglevel,
                flat_child= flat_child)
        self._log = logger

        self._log.info(f'*** MOTorch : {name} *** initializes..')
        self._log.info(f'> {name} save_topdir: {save_topdir}{" <- read only mode!" if _read_only else ""}')

        # init as a ParaSave
        super().__init__(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx,
            logger=         get_child(self._log, 'ParaSave_logger'),
            **kwargs)
        point_saved = self.get_point()

        # **************************************************************************************** further resolve POINT

        ### resolve module_type

        module_type_saved = point_saved.get('module_type', None)

        if not module_type and not module_type_saved:
            msg = 'module_type was not given and has not been found in saved, cannot continue!'
            self._log.error(msg)
            raise MOTorchException(msg)

        if module_type and module_type_saved and module_type != module_type_saved:
            self._log.info('given module_type differs from module_type found in saved, using saved')

        module_type = module_type_saved or module_type
        self._log.info(f'> {self.name} module_type: {module_type.__name__}')

        ### manage params from self.module_type.__init__

        _module_init_def = get_class_init_params(module_type)['with_defaults'] # defaults of self.module_type.__init__

        """
        # special case of params: [device, dtype] <- those will be set with values prepared by MOTorch below, BUT...
        _override_in_module_for_none = {
            'device':   self.MOTORCH_DEFAULTS['device'],
            'dtype':    self.MOTORCH_DEFAULTS['dtype']}

        # ..EXCEPT a case when are set in self.module_type.__init__ to other value than None
        remove_from_override = []
        for param in _override_in_module_for_none:
            if param in _module_init_def and _module_init_def[param] is not None:
                remove_from_override.append(param)
        for param in remove_from_override:
            _override_in_module_for_none.pop(param)
        """

        ### update in proper order

        self._point = {}
        self._point.update(ParaSave.PARASAVE_DEFAULTS)
        self._point.update(self.MOTORCH_DEFAULTS)
        self._point.update(_module_init_def)
        #self._point.update(_override_in_module_for_none)
        self._point.update(point_saved)
        self._point.update(kwargs)
        self._point["module_type"] = module_type

        # remove logger (may come from Module init defaults)
        if 'logger' in self._point:
            self._point.pop('logger')

        ### finally resolve device

        # device parameter, may be given to MOTorch in DevicesTorchness type
        # it needs to be cast to PyTorch namespace here
        self._log.debug(f'> {self.name} resolves devices, given: {self._point["device"]}')
        self._log.debug(f'> torch.cuda.is_available(): {torch.cuda.is_available()}')
        devices = get_devices(
            devices=            self._point["device"],
            torch_namespace=    True,
            logger=             get_child(self._log, 'get_devices'))
        if not devices:
            self._log.warning(f'given device: {self._point["device"]} is not available, using CPU')
            devices = ['cpu']
        device = devices[0]
        self._log.info(f'> {self.name} given devices: {self._point["device"]}, will use: {device}')
        self._point['device'] = device

        ### prepare Module point and extract not used kwargs

        self._module_point = point_trim(module_type, self._point)
        self._module_point['logger'] = get_child(self._log, 'Module_logger')

        ### report

        self._log.debug(f'{self.name} POINT sources:')
        self._log.debug(f'> PARASAVE_DEFAULTS:        {ParaSave.PARASAVE_DEFAULTS}')
        self._log.debug(f'> MOTORCH_DEFAULTS:         {self.MOTORCH_DEFAULTS}')
        self._log.debug(f'> Module.__init__ defaults: {_module_init_def}')
        self._log.debug(f'> POINT saved:              {point_saved}')
        self._log.debug(f'> given kwargs:             {kwargs}')
        self._log.debug(f'Module complete POINT:      {self._module_point}')
        self._log.debug(f'MOTorch complete POINT:     {self._point}')

        _kwargs_not_used = {}
        out = get_class_init_params(MOTorch)
        motorch_init_params = out['without_defaults'] + list(out['with_defaults'].keys())
        motorch_params_all = list(ParaSave.PARASAVE_DEFAULTS.keys()) + list(self.MOTORCH_DEFAULTS.keys()) + motorch_init_params
        for k in kwargs:
            if k not in self._module_point and k not in motorch_params_all:
                _kwargs_not_used[k] = kwargs[k]
        if _kwargs_not_used:
            self._log.warning(f'> there are kwargs given but not used by MOTorch nor Module: {_kwargs_not_used}')

        self.update(self._point)

        # parameters names safety check
        found = self.check_params_sim(params=list(self.MOTORCH_DEFAULTS.keys()) + list(kwargs.keys()))
        if found:
            self._log.warning(f'{self.name} (MOTorch) was asked to check for params similarity and found:')
            for pa, pb in found:
                self._log.warning(f'> params \'{pa}\' and \'{pb}\' are close !!!')

        # set seed in all possible areas (https://pytorch.org/docs/stable/notes/randomness.html)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        ### build MOTorch Module

        self._log.info(f'{self.name} builds graph of {self.module_type.__name__}')
        self._module = self.module_type(**self._module_point)

        if self.try_load_ckpt:
            self.load_ckpt()
        else:
            self._log.info(f'> {self.name} checkpoint not loaded, not even tried because \'try_load_ckpt\' was set to {self.try_load_ckpt}')

        self._log.debug(f'> moving {self.name} to device: {self.device}, dtype: {self.dtype}')
        self._module.to(self.device)
        self._module.to(self.dtype)

        self._log.debug(f'{self.name} Module initialized!')

        ### resolve optimizer

        opt_kwargs = {}
        try:
            self.opt_class, opt_kwargs = self._module.get_optimizer_def()
            self._log.debug(f'using optimizer from Module: {self.opt_class.__name__}, Module optimizer kwargs: {opt_kwargs}')
        except MOTorchException as e:
            self._log.debug(f'using optimizer resolved by MOTorch: {self.opt_class.__name__}')

        opt_kwargs['params'] = self._module.parameters()
        opt_kwargs['lr'] = self.baseLR
        self._opt = self.opt_class(**opt_kwargs)
        self._log.debug(f'MOTorch optimizer:\n{self._opt}')

        # from now LR is managed by scheduler
        self._scheduler = ScaledLR(
            optimizer=      self._opt,
            starting_step=  self.train_step,
            warm_up=        self.warm_up,
            n_wup_off=      self.n_wup_off,
            ann_base=       self.ann_base,
            ann_step=       self.ann_step,
            logger=         get_child(self._log, 'ScaledLR'))

        self._grad_clipper = GradClipperMAVG(
            do_clip=        self.gc_do_clip,
            module=         self._module,
            start_val=      self.gc_start_val,
            factor=         self.gc_factor,
            first_avg=      self.gc_first_avg,
            max_clip=       self.gc_max_clip,
            max_upd=        self.gc_max_upd,
            logger=         get_child(self._log, 'GradClipperMAVG'))

        # MOTorch by default is not in training mode
        self.train(False)
        self._log.debug(f'> set {self.name} train.mode to False..')

        # TensorBoard writer
        self._TBwr = tbwr or TBwr(
            logdir=self._get_model_dir(
                model_name=     self.name,
                save_topdir=    self.save_topdir)) if self.do_TB else None

        self._batcher = None

        self._log.debug(str(self))
        self._log.info(f'MOTorch init finished!')

    @classmethod
    def _get_name(
            cls,
            module_type: Optional[type(Module)]=    None,
            name: Optional[str]=                    None,
            name_timestamp=                         False,
    ) -> str:
        """ resolves MOTorch name """
        # resolve name
        if not name:
            name = f'{module_type.__name__}_{cls.__name__}'
        if name_timestamp:
            name += f'_{stamp()}'
        return name

    # **************************************************************************** model call (run NN with data) methods

    def __call__(
            self,
            *args,
            bypass_data_conv=               False,
            set_training: Optional[bool]=   None,   # for dropout etc
            no_grad=                        True,   # by default gradients calculation is disabled for FWD call
            empty_cuda_cache=               True,   # releases all unoccupied cached memory after model call currently held by the caching allocator
            **kwargs,
    ) -> DTNS:
        """ forward (FWD) call
        runs forward on nn.Module, manages:
        - data type / format preparation
        - training mode
        - gradients computation
        """

        if set_training is not None:
            self.train(set_training)

        if not (bypass_data_conv or self.bypass_data_conv):
            args = [self.convert(data=a) for a in args]
            kwargs = {k: self.convert(data=kwargs[k]) for k in kwargs}

        if no_grad:
            with torch.no_grad():
                out = self._module(*args, **kwargs)
        else:
            out = self._module(*args, **kwargs)

        # eventually roll back to MOTorch default
        if set_training:
            self.train(False)

        if empty_cuda_cache:
            torch.cuda.empty_cache()

        return out


    def convert(self, data:Any) -> TNS:
        """ converts given data to torch.Tensor compatible with self (device,dtype) """

        # do not convert None
        if type(data) is not None:

            if type(data) is not torch.Tensor:
                if type(data) is np.ndarray: data = torch.from_numpy(data)
                else:                        data = torch.tensor(data)

            # convert device + float types
            data = data.to(self.device, self.dtype if data.is_floating_point() or data.is_complex() else None)

        return data


    def loss(
            self,
            *args,
            bypass_data_conv=               False,
            set_training: Optional[bool]=   None,   # for not None forces given training mode for torch.nn.Module
            no_grad=                        False,  # by default gradients calculation is enabled for loss call
            empty_cuda_cache=               True,   # releases all unoccupied cached memory after model call currently held by the caching allocator
            **kwargs,
    ) -> DTNS:
        """ forward + loss call on NN """

        if set_training is not None:
            self.train(set_training)

        if not (bypass_data_conv or self.bypass_data_conv):
            args = [self.convert(data=a) for a in args]
            kwargs = {k: self.convert(data=kwargs[k]) for k in kwargs}

        if no_grad:
            with torch.no_grad():
                out = self._module.loss(*args, **kwargs)
        else:
            out = self._module.loss(*args, **kwargs)

        # eventually roll back to MOTorch default
        if set_training:
            self.train(False)

        if empty_cuda_cache:
            torch.cuda.empty_cache()

        return out


    def backward(
            self,
            *args,
            bypass_data_conv=   False,
            set_training: bool= True,   # for backward training mode is set to True by default
            empty_cuda_cache=   True,   # releases all unoccupied cached memory after model call currently held by the caching allocator
            **kwargs
    ) -> DTNS:
        """ backward call on NN, runs loss calculation + update of Module """

        out = self.loss(
            *args,
            bypass_data_conv=   bypass_data_conv,
            set_training=       set_training,
            no_grad=            False, # True makes no sense with backward()
            empty_cuda_cache=   empty_cuda_cache,
            **kwargs)

        self._opt.zero_grad()               # clear gradients
        out['loss'].backward()              # build gradients
        gnD = self._grad_clipper.clip()     # clip gradients, adds: 'gg_norm' & 'gg_norm_clip' to out
        self._opt.step()                    # apply optimizer
        self._scheduler.step()              # apply LR scheduler
        self.train_step += 1                # update step

        out['currentLR'] = self._scheduler.get_last_lr()[0] # INFO: currentLR of the first group is taken
        out.update(gnD)

        if empty_cuda_cache:
            torch.cuda.empty_cache()

        return out

    # *********************************************************************************************** load / save / copy

    @classmethod
    def _get_model_dir(cls, model_name:str, save_topdir:Optional[str]=None) -> str:
        """ returns model directory path """
        if not save_topdir: save_topdir = cls.SAVE_TOPDIR
        return f'{save_topdir}/{model_name}'

    @classmethod
    def _get_ckpt_path(cls, model_name:str, save_topdir:Optional[str]=None) -> str:
        """ returns path of checkpoint pickle file """
        model_dir = cls._get_model_dir(model_name=model_name, save_topdir=save_topdir)
        return f'{model_dir}/{model_name}.pt'


    def load_ckpt(
            self,
            name: Optional[str]=        None,  # allows to load custom name (model_name)
            save_topdir: Optional[str]= None,  # allows to load from custom save_topdir
    ) -> Optional[dict]:
        """ tries to load checkpoint and return additional data """

        ckpt_path = self._get_ckpt_path(
            model_name=     name or self.name,
            save_topdir=    save_topdir or self.save_topdir)

        save_obj = None

        try:
            save_obj = torch.load(f=ckpt_path, map_location=self.device) # immediately place all tensors to current device (not previously saved one)
            self._module.load_state_dict(save_obj.pop('model_state_dict'))
            self._log.info(f'> {self.name} checkpoint loaded from {ckpt_path}')
        except Exception as e:
            # this exception logs as INFO since it is quite normal to not load checkpoint while init
            self._log.info(f'> {self.name} checkpoint NOT loaded because of exception: {e}')

        return save_obj


    def save_ckpt(
            self,
            name: Optional[str]=                None,   # allows to save under custom name (model_name)
            save_topdir: Optional[str]=         None,   # allows to save in custom save_topdir
            additional_data: Optional[Dict]=    None,   # allows to save additional
    ) -> None:
        """ saves model checkpoint & optionally additional data """

        ckpt_path = self._get_ckpt_path(
            model_name=     name or self.name,
            save_topdir=    save_topdir or self.save_topdir)

        save_obj = {'model_state_dict': self._module.state_dict()}
        if additional_data: save_obj.update(additional_data)

        torch.save(obj=save_obj, f=ckpt_path)


    def save(self):
        """ saves MOTorch (ParaSave POINT and model checkpoint) """

        if self.read_only:
            raise MOTorchException('read_only MOTorch cannot be saved!')

        # to properly start grad clipping after load
        self['gc_first_avg'] = False
        self['gc_start_val'] = float(self._grad_clipper.gg_norm_clip)

        self.save_point()
        self.save_ckpt()
        self._log.info(f'{self.__class__.__name__} {self.name} saved to {self.save_topdir}')

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
            logger=                         None,
            loglevel=                       30):
        """ copies full MOTorch folder (POINT & checkpoints) """

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
            loglevel=           loglevel)

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
        """ GX on 2 checkpoints only of saved 2 MOTorch """

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
        """ performs GX on saved MOTorch """

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

    # ***************************************************************************************************** train / test

    def load_data(
            self,
            data_TR: Dict[str,np.ndarray],
            data_TS: Optional[Union[Dict[str,NPL], Dict[str,Dict[str,NPL]]]]=   None,
            split_factor: float=                                                0.0):
        """ converts and loads data to Batcher """

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
            logger=         get_child(self._log, 'Batcher'))


    def run_train(
            self,
            data_TR: Dict[str,np.ndarray],  # INFO: it also accepts Dict[str,torch.Tensor]
            data_TS: Optional[Union[Dict[str,NPL], Dict[str,Dict[str,NPL]]]]=   None,
            split_factor: float=        0.0,
            n_batches: Optional[int]=   None,
            test_freq=                  100,    # number of batches between tests, model SHOULD BE tested while training
            mov_avg_factor=             0.1,
            save_max=                   True,   # allows to save model while training (after max test)
            use_F1=                     True,   # uses F1 as a train/test score (not acc)
        ) -> Optional[float]:
        """ trains model, returns optional test score """

        if data_TR:
            self.load_data(data_TR=data_TR, data_TS=data_TS, split_factor=split_factor)

        if not self._batcher: raise MOTorchException(f'{self.name} has not been given data for training, use load_data()')

        self._log.info(f'{self.name} - training starts [acc / F1 / loss]')
        self._log.info(f'data sizes (TR,VL,TS) samples: {self._batcher.get_data_size()}')

        if n_batches is None: n_batches = self.n_batches  # take default
        self._log.info(f'batch size:             {self["batch_size"]}')
        self._log.info(f'train for num_batches:  {n_batches}')

        batch_IX = 0                            # this loop (local) batch counter
        tr_accL = []
        tr_f1L = []
        tr_lssL = []

        score_name = 'F1' if use_F1 else 'acc'
        ts_score_max = 0                        # test score (acc or F1) max
        ts_score_all_results = []               # test score all results
        ts_score_mav = MovAvg(mov_avg_factor)   # test score (acc or F1) moving average

        # initial save
        if not self.read_only and save_max:
            self.save_ckpt()

        ts_bIX = [bIX for bIX in range(n_batches+1) if not bIX % test_freq] # batch indexes when test will be performed
        assert ts_bIX, 'ERR: model SHOULD BE tested while training!'
        ten_factor = int(0.1*len(ts_bIX)) # number of tests for last 10% of training
        if ten_factor < 1: ten_factor = 1 # we need at least one result
        if self.hpmser_mode: ts_bIX = ts_bIX[-ten_factor:]

        while batch_IX < n_batches:

            out = self.backward(**self._batcher.get_batch(), bypass_data_conv=True)

            loss = out['loss']
            acc = out['acc'] if 'acc' in out else None
            f1 = out['f1'] if 'f1' in out else None

            batch_IX += 1

            if self.do_TB:
                self.log_TB(value=loss,                tag='tr/loss',    step=self.train_step)
                self.log_TB(value=out['gg_norm'],      tag='tr/gn',      step=self.train_step)
                self.log_TB(value=out['gg_norm_clip'], tag='tr/gn_clip', step=self.train_step)
                self.log_TB(value=out['currentLR'],    tag='tr/cLR',     step=self.train_step)
                if acc is not None:
                    self.log_TB(value=acc,             tag='tr/acc',     step=self.train_step)
                if f1 is not None:
                    self.log_TB(value=f1,              tag='tr/F1',      step=self.train_step)

            if acc is not None: tr_accL.append(acc)
            if f1 is not None: tr_f1L.append(f1)
            tr_lssL.append(loss)

            if batch_IX in ts_bIX:

                res = self.run_test()
                first_key = list(res.keys())[0]

                for k in res:

                    ts_loss, ts_acc, ts_f1 = res[k]

                    ts_score = ts_f1 if use_F1 else ts_acc
                    if ts_score is not None:
                        ts_score_all_results.append(ts_score)

                    key_name = f'_{k}' if k is not None else ''
                    if self.do_TB:
                        if ts_loss is not None:
                            self.log_TB(value=ts_loss,                    tag=f'ts{key_name}/loss',              step=self.train_step)
                        if ts_acc is not None:
                            self.log_TB(value=ts_acc,                     tag=f'ts{key_name}/acc',               step=self.train_step)
                        if ts_f1 is not None:
                            self.log_TB(value=ts_f1,                      tag=f'ts{key_name}/F1',                step=self.train_step)
                        if ts_score is not None:
                            self.log_TB(value=ts_score_mav.upd(ts_score), tag=f'ts{key_name}/{score_name}_mav', step=self.train_step)

                    tr_acc_nfo = f'{100*sum(tr_accL)/test_freq:.1f}' if acc is not None else '--'
                    tr_f1_nfo =  f'{100*sum(tr_f1L)/test_freq:.1f}' if f1 is not None else '--'
                    tr_loss_nfo = f'{sum(tr_lssL)/test_freq:.3f}'
                    ts_acc_nfo = f'{100*ts_acc:.1f}' if ts_acc is not None else '--'
                    ts_f1_nfo = f'{100*ts_f1:.1f}' if ts_f1 is not None else '--'
                    ts_loss_nfo = f'{ts_loss:.3f}' if ts_loss is not None else '--'
                    self._log.info(f'# {self["train_step"]:5d} TR: {tr_acc_nfo} / {tr_f1_nfo} / {tr_loss_nfo} -- TS{key_name}: {ts_acc_nfo} / {ts_f1_nfo} / {ts_loss_nfo}')
                    tr_accL = []
                    tr_f1L = []
                    tr_lssL = []

                    # model is saved for max ts_score for the first_key (TS name)
                    if k==first_key and ts_score is not None and ts_score > ts_score_max:
                        ts_score_max = ts_score
                        if not self.read_only and save_max:
                            self.save_ckpt()

        self._log.info(f'### model {self.name} finished training')

        ts_score_fin = None

        if save_max:
            ts_score_fin = ts_score_max
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
            self._log.info(f' > test_{score_name}_max: {ts_score_max:.4f}')
            self._log.info(f' > test_{score_name}_fin: {ts_score_fin:.4f}')
            if self.do_TB:
                self.log_TB(value=ts_score_fin, tag=f'ts/ts_{score_name}_fin', step=self.train_step)

        return ts_score_fin


    def run_test(
            self,
            data: Optional[Dict[str,np.ndarray]]=   None,
            split_factor: float=                    1.0, # if data for test will be given above, by default MOTorch will be tested on ALL
    ) -> Dict[Union[str,None], Tuple[Optional[float], Optional[float], Optional[float]]]:
        """ tests model
        returns:
            - optional loss (average)
            - optional accuracy, optional F1
            - optional loss <- since there may be not TS batches
        """

        if data:
            self.load_data(data_TR=data, split_factor=split_factor)

        if not self._batcher:
            raise MOTorchException(f'{self.name} has not been given data for testing, use load_data() or give it while testing!')

        res = {}
        for k in self._batcher.get_TS_names():

            batches = self._batcher.get_TS_batches()
            lossL = []
            accL = []
            f1L = []
            n_all = 0
            for batch in batches:
                out = self.loss(**batch, bypass_data_conv=True)
                n_new = len(out['logits'])
                n_all += n_new
                lossL.append(out['loss']*n_new)
                if 'acc' in out: accL.append(out['acc']*n_new)
                if 'f1' in out:  f1L.append(out['f1']*n_new)

            acc_avg = sum(accL)/n_all if accL else None
            f1_avg = sum(f1L)/n_all if f1L else None
            loss_avg = sum(lossL)/n_all if lossL else None
            res[k] = loss_avg, acc_avg, f1_avg

        return res

    # *********************************************************************************************** other / properties

    def update_baseLR(self, lr: float):
        """ updates scheduler baseLR of 0 group """
        self.baseLR = lr
        self._scheduler.update_base_lr0(lr)

    @property
    def module(self):
        return self._module

    def train(self, mode:bool=True):

        return self._module.train(mode)

    @property
    def tbwr(self):
        return self._TBwr

    def log_TB(self, value, tag:str, step:Optional[int]=None):
        """ logs value to TB """
        if step is None:
            step = self.train_step
        if self.do_TB:
            self._TBwr.add(value=value, tag=tag, step=step)
        else: self._log.warning(f'{self.name} cannot log to TensorBoard since \'do_TB\' flag was set to False!')

    def log_histogram_TB(self, values, tag:str, step:Optional[int]=None, bins="tensorflow"):
        """ logs values to TB histogram """
        if step is None:
            step = self.train_step
        if self.do_TB:
            self._TBwr.add_histogram(values=values, tag=tag, step=step, bins=bins)
        else: self._log.warning(f'{self.name} cannot log to TensorBoard since \'do_TB\' flag was set to False!')

    @property
    def logger(self):
        return self._log

    @property
    def size(self) -> int:
        return sum([p.numel() for p in self._module.parameters()])

    def __str__(self):
        s = f'{self.__class__.__name__} (MOTorch): {ParaSave.__str__(self)}\n'
        s += f'{str(self._module)}\n ### model size: {self.size} params'
        return s