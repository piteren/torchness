import numpy as np
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.printout import nice_scin
import torch
from typing import Optional, List


class ScaledLR(torch.optim.lr_scheduler.LRScheduler):
    """ Applies warm-up and annealing for LR of 0 group.
    ScaledLR.step() should be called every update / batch / step """

    def __init__(
            self,
            optimizer,
            step: int=                      0,      # current step (to start with)
            warmup_end: int=                1000,   # warm-up starts at 0 step and goes for warmup_end steps
            anneal_start: Optional[int]=    10_000, # None turns off annealing
            anneal_base: float=             0.999,  # 1.0 turns off annealing, lower values speed-up
            anneal_mul: float=              1.0,    # higher values speed-up annealing
            last_epoch=                     -1,
            logger=                         None,
            loglevel=                       20,
    ):

        self.logger = logger or get_pylogger(name='ScaledLR', level=loglevel)

        self._step = step
        self.w_end = warmup_end
        self.a_start = anneal_start
        self.a_base = anneal_base
        self.a_mul = anneal_mul

        super(ScaledLR, self).__init__(optimizer, last_epoch)

    def update_base_lr0(self, lr: float):
        """ updates LR of group 0 """
        self.base_lrs[0] = lr

    def get_lr(self) -> List[float]:

        lrs = np.asarray(self.base_lrs) # self.base_lrs is a list that keeps baseLR of groups
        if self.w_end and self._step < self.w_end:
            w_ratio = self._step / self.w_end
            lrs *= w_ratio
            self.logger.debug(f'current warm-up ratio:{w_ratio}')

        if self.a_start is not None and self.a_base != 1.0:
            a_steps = max(0, self._step - self.a_start)
            if a_steps > 0:
                factor = self.a_base ** (a_steps * self.a_mul)
                lrs *= factor
                self.logger.debug(f'current annealing factor:{nice_scin(factor)}')

        self.logger.debug(f'ScaledLR scheduler step:{self._step}, resulting LR:{nice_scin(lrs[0])}')
        return lrs.tolist()

    def step(self, epoch:Optional[int]=None):
        super(ScaledLR, self).step(epoch)
        self._step += 1