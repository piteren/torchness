from pypaq.lipytools.pylogger import get_pylogger
import torch
from typing import Optional, Dict

from torchness.tools import count_model_params


class NNModule(torch.nn.Module):
    """ Wraps torch.nn.Module with some additional tools.
    It is a much simpler version  of NN management class than MOTorch, useful in smaller projects.

    NNModule implements:
    - manages seed
    - inits / manages logger
    - tracks kwargs_not_used
    - stores / manages NN model params (hyperparameters)
    - saves / builds NN Model with ckpt and params

    HOW TO USE:

    1. override __init__():
        a. pass all model params passed directly to it
        b. pass seed, device, logger, loglevel
        c. add **kwargs_not_used for monitoring kwargs given to init, which are in fact not used by your module
        d. call super().__init__() with params, seed, device, logger, loglevel, kwargs_not_used
        example:

        def __init__(
                self,
                    **here go params, like:
                        d_model: Optional[int]= None,
                        num_layers: int=        3,
                seed=       123,
                device=     'cuda',
                logger=     None,
                loglevel=   20,
                **kwargs_not_used,
        ):
            super().__init__(
                d_model=            d_model,
                num_layers=         num_layers,
                seed=               seed,
                device=             device,
                logger=             logger,
                loglevel=           loglevel,
                kwargs_not_used=    kwargs_not_used)

            **here bild a module in self

            self.to(self.device)

    2. implement forward() and other methods needed """

    def __init__(
            self,
            seed=                               123,
            device=                             'cuda',
            logger=                             None,
            loglevel=                           20,
            kwargs_not_used: Optional[Dict]=    None,
            **params):
        """
        :param seed:
        :param device:
        :param logger:
        :param loglevel:
        :param kwargs_not_used:
            put here a dict of not used kwargs
            -> a good practice to track them
            -> only to be logged
        :param params:
            put here every NN hyperparameter """

        super().__init__()

        if not logger:
            logger = self.get_logger(loglevel)
        self.logger = logger
        self.logger.info(f'*** NNModule ({self.__class__.__name__}) *** initializes ..')

        self.params = params
        self.params['seed'] = seed
        self.device = device

        for p,pv in self.params.items():
            self.logger.info(f'> {p:20}: {pv}')
        if kwargs_not_used:
            self.logger.info(f'>> kwargs_not_used: {kwargs_not_used}')

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, **kwargs):
        """ forward pass (FWD) method, to be implemented """
        raise NotImplementedError

    @classmethod
    def get_logger(cls, loglevel=20):
        return get_pylogger(name=cls.__name__, add_stamp=True, level=loglevel)

    def __str__(self):
        return (f'{self.__class__.__name__} (#{count_model_params(self)})\n'
                f'> params: {self.params}\n'
                f'{super().__str__()}')

    def disable_grad(self, pattern:str):
        for n,p in  self.named_parameters():
            if pattern in n:
                self.logger.debug(f'disabled grad for {n}')
                p.requires_grad = False

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def parameters_nfo(self) -> str:
        trainable_size = 0
        non_trainable_size = 0
        n_trainable = 0
        n_not_trainable = 0
        s = f'{"---name---":50}{"---shape---":40}{"---trainable---"}\n'
        for n,p in self.named_parameters():
            p_size = p.numel()
            if p.requires_grad:
                trainable_size += p_size
                n_trainable += 1
            else:
                non_trainable_size += p_size
                n_not_trainable += 1
            s += f'{n:50}{str(p.shape):40}{p.requires_grad}\n'
        tot = trainable_size + non_trainable_size
        s += f' # ___TOT:        ({n_trainable+n_not_trainable:02}) {tot}\n'
        s += f' # trainable:     ({n_trainable:02}) {trainable_size} ({trainable_size/tot*100:.1f}%)\n'
        s += f' # non trainable: ({n_not_trainable:02}) {non_trainable_size} ({non_trainable_size/tot*100:.1f}%)'
        return s

    def save(self, ckpt_fp:str):
        torch.save(obj={"model":self.state_dict(), "params":self.params}, f=ckpt_fp)

    @classmethod
    def build(
            cls,
            ckpt_fp: Optional[str]= None,
            device=                 'cuda',
            logger=                 None,
            loglevel=               20,
            **kwargs):
        """ Builds class object from a given ckpt or defaults.
        Object parameters should be saved with ckpt in 'params'.

        For backward compatibility, it allows to put params with kwargs
        to override defaults of cls.__init__().
        After saving object (with obj.save()),
        overriding with kwargs will be no longer needed """

        if not logger:
            logger = cls.get_logger(loglevel)

        logger.info(f'building {cls.__name__} (NNModule) from ckpt: {ckpt_fp}')
        logger.info(f'> device: {device}')

        module_kwargs = {}
        ckpt = None
        if ckpt_fp:
            ckpt = torch.load(f=ckpt_fp, map_location=device, weights_only=True)
            logger.info(f'> ckpt got: {list(ckpt.keys())}')
            if 'params' in ckpt:
                ckpt_pms = ckpt['params']
                logger.info(f'> params from ckpt: {ckpt_pms}')
                module_kwargs.update(ckpt_pms)

        if kwargs:
            logger.info(f'> given kwargs: {kwargs}')
            module_kwargs.update(kwargs)
        module_kwargs['device'] = device
        module_kwargs['logger'] = logger

        net = cls(**module_kwargs)

        if ckpt:
            net.load_state_dict(ckpt['model'], strict=True)

        return net