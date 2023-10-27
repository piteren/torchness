from pypaq.lipytools.moving_average import MovAvg
from pypaq.pytypes import NUM, NPL
import torch
from typing import Optional


def clip_grad_norm_(
        parameters: NPL,
        max_norm: NUM,
        norm_type: NUM= 2.0,
        do_clip: bool=  True, # disables clipping (just GN calculations)
) -> NUM:
    """ clips (scales) gradients of given parameters
    copied & refactored from torch.nn.utils.clip_grad.py
    returns norm of original parameters
    """

    if isinstance(parameters, torch.Tensor): parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0: return 0.0

    device = parameters[0].grad.device
    if norm_type == torch.inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)

    if do_clip:
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for p in parameters:
            p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))

    return total_norm


class GradClipperMAVG:
    """ clips gradients of parameters of given Module
    with value:
    - given OR
    - updated with MovAvg
    """

    def __init__(
            self,
            module: torch.nn.Module,
            clip_value: Optional[float]=    None,   # clipping value, for None clips with mavg
            factor: NUM=                    0.01,   # MovAvg factor
            first_avg=                      True,   # use averaging @start
            start_val: NUM=                 0.1,    # use this value @start (for first clip..)
            max_val: Optional[NUM]=         None,   # max value of gg_mavg
            max_upd: NUM=                   1.5,    # max factor of gg_mavg to update with
            do_clip: bool=                  True):  # disables clipping (just GN calculations)

        self.module = module
        self.clip_value = clip_value

        self._gg_norm_mavg = MovAvg(factor=factor, first_avg=first_avg)
        self._gg_norm_mavg.upd(start_val)
        self._gg_norm_mavg_max = max_val

        self.max_upd = max_upd
        self.do_clip = do_clip

    # clip & update parameters
    def clip(self):

        gg_norm_mavg = self._gg_norm_mavg()
        if self._gg_norm_mavg_max and gg_norm_mavg > self._gg_norm_mavg_max:
            gg_norm_mavg = self._gg_norm_mavg_max

        gg_norm = clip_grad_norm_(
            parameters= self.module.parameters(),
            max_norm=   self.clip_value or gg_norm_mavg,
            do_clip=    self.do_clip)

        avt_update = min(gg_norm, self.max_upd * self._gg_norm_mavg()) # max value of update
        gg_norm_clip = self._gg_norm_mavg.upd(avt_update)

        return {
            'gg_norm':      gg_norm,
            'gg_norm_clip': gg_norm_clip}

    @property
    def gg_norm_clip(self):
        return self._gg_norm_mavg()