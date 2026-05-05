import numpy as np
from pypaq.lipytools.pylogger import Logged
from pypaq.lipytools.printout import nice_scin
import torch


class ScaledLR(torch.optim.lr_scheduler.LRScheduler, Logged):
    """ Applies warm-up and annealing for LR of 0 group.
    ScaledLR.step() should be called every update / batch / step """

    def __init__(
            self,
            optimizer,
            step: int = 0,
            warmup_end: int | None = 1000,
            anneal_start: int | None = 10_000,
            anneal_base: float = 0.999,
            anneal_mul: float = 1.0,
            last_epoch = -1,
            loglevel: int = 20,
    ):
        self.logger = self.get_logger(level=loglevel)

        self._step = step
        self.w_end = warmup_end
        self.a_start = anneal_start
        self.a_base = anneal_base
        self.a_mul = anneal_mul

        super(ScaledLR, self).__init__(optimizer, last_epoch)

    def update_base_lr0(self, lr: float):
        """ updates LR of group 0 """
        self.base_lrs[0] = lr

    def get_lr(self) -> list[float]:

        lrs = np.asarray(self.base_lrs)
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

    def step(self, epoch: int | None = None):
        super(ScaledLR, self).step(epoch)
        self._step += 1