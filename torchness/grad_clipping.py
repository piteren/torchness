from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.pylogger import get_pylogger
import torch
from typing import Optional, Union, Iterator

from torchness.base import TNS, NUM, NPL, TorchnessException


def clip_grad_norm_(
        parameters: Union[NPL, Iterator[TNS]],
        norm_type: NUM=             2.0,
        max_norm: Optional[NUM]=    None,
        do_clip: bool=              True,  # disables clipping (just GN calculations)
) -> NUM:
    """ computes and returns gradients norm (input) of given parameters,
    then optionally clips (scales) gradients,
    copied & refactored from torch.nn.utils.clip_grad.py """

    # filter out parameters
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters_grad = [p for p in parameters if p.grad is not None]
    if len(parameters_grad) == 0:
        return 0.0

    device = parameters_grad[0].grad.device  # choose single device for computation
    if norm_type == torch.inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters_grad]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), norm_type).to(device)
                for p in parameters_grad]),
            norm_type)

    if do_clip:
        if max_norm is None:
            raise TorchnessException('max_norm must be given when clipping')
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for p in parameters_grad:
            p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))

    return total_norm


class GradClipperMAVG:
    """ clips gradients of parameters of given Module with MovAvg value """

    def __init__(
            self,
            module: torch.nn.Module,
            start_val: NUM=             0.1,    # MovAvg start value
            factor: NUM=                0.01,   # MovAvg factor
            first_avg=                  True,   # use MovAvg start averaging
            max_clip: Optional[NUM]=    None,   # clipped value won't go higher
            max_upd: NUM=               1.5,    # max factor of gg_mavg to update with
            do_clip: bool=              True,   # disables clipping (just GN calculations)
            logger=                     None,
            loglevel=                   20,
    ):
        if not logger:
            logger = get_pylogger(name='GradClipperMAVG', level=loglevel)
        self.logger = logger

        self.module = module

        self.mavg = MovAvg(factor=factor, first_avg=first_avg)
        self.mavg.upd(start_val)
        self.max_clip = max_clip

        self.max_upd = max_upd
        self.do_clip = do_clip

    # clip & update parameters
    def clip(self):

        gg_norm_clip = self.gg_norm_clip
        self.logger.debug(f'gg_norm_clip: {gg_norm_clip}')

        gg_norm = clip_grad_norm_(
            parameters= self.module.parameters(),
            max_norm=   gg_norm_clip,
            do_clip=    self.do_clip)
        self.logger.debug(f'gg_norm: {gg_norm}')

        mavg_update = min(gg_norm, gg_norm_clip*self.max_upd)
        if self.max_clip and mavg_update > self.max_clip:
            mavg_update = self.max_clip
        self.mavg.upd(mavg_update)

        return {
            'gg_norm':      gg_norm,
            'gg_norm_clip': gg_norm_clip}

    @property
    def gg_norm_clip(self):
        return self.mavg()