from collections.abc import Iterable
import logging
from pypaq.lipytools.moving_average import MovAvg
import torch

from torchness.base import TNS, NUM, NPL, TorchnessException

logger = logging.getLogger(__name__)


def clip_grad_norm_(
        parameters: NPL | Iterable[TNS],
        max_norm: NUM | None = None,
        norm_type: NUM = 2.0,
        do_clip: bool = True,
) -> float:
    """ computes and returns gradients norm (input) of given parameters,
    then optionally clips (scales) gradients,
    copied & refactored from torch.nn.utils.clip_grad.py """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters_grad = [p for p in parameters if p.grad is not None]
    if not parameters_grad:
        return 0.0

    device = parameters_grad[0].grad.device
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

    return total_norm.item()


class GradClipperMAVG:
    """ clips gradients of parameters of given Module with MovAvg value """

    def __init__(
            self,
            module: torch.nn.Module,
            start_val: NUM = 0.1,
            factor: NUM = 0.01,
            first_avg = True,
            max_clip: NUM | None = None,
            max_upd: NUM = 1.5,
            do_clip: bool = True,
    ):
        self.module = module

        self.mavg = MovAvg(factor=factor, first_avg=first_avg)
        self.mavg.upd(start_val)
        self.max_clip = max_clip

        self.max_upd = max_upd
        self.do_clip = do_clip

    def clip(self) -> dict[str, float]:

        gg_norm_clip = self.mavg()
        logger.debug(f'gg_norm_clip: {gg_norm_clip}')

        gg_norm = clip_grad_norm_(
            parameters= self.module.parameters(),
            max_norm=   gg_norm_clip,
            do_clip=    self.do_clip)
        logger.debug(f'gg_norm: {gg_norm}')

        mavg_update = min(gg_norm, gg_norm_clip*self.max_upd)
        if self.max_clip:
            mavg_update = min(mavg_update, self.max_clip)
        self.mavg.upd(mavg_update)

        return {'gg_norm': gg_norm, 'gg_norm_clip': gg_norm_clip}