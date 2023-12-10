from torch.utils.tensorboard import SummaryWriter
from typing import Optional



class TBwr:
    """ TensorBoard writer based on PyTorch wraps of TensorBoard """

    def __init__(
            self,
            logdir: str,
            flush_secs= 10):
        self.logdir = logdir
        self.flush_secs = flush_secs
        # INFO: SummaryWriter creates logdir while init, because of that self.sw init has moved here (in the first call of add)
        self.sw = None
        self.step = {} # step per tag

    def _get_sw(self):
        return SummaryWriter(
            log_dir=    self.logdir,
            flush_secs= self.flush_secs)

    def _get_step(self, tag:str):
        if tag not in self.step:
            self.step[tag] = 0
        step = self.step[tag]
        self.step[tag] += 1
        return step

    def add(self,
            value,
            tag: str,
            step: Optional[int]=    None):
        if not self.sw:
            self.sw = self._get_sw()
        if step is None:
            step = self._get_step(tag)
        self.sw.add_scalar(
            tag=            tag,
            scalar_value=   value,
            global_step=    step)

    def add_histogram(
            self,
            values,
            tag: str,
            step: Optional[int]=    None,
            bins=                   "tensorflow"):
        if not self.sw:
            self.sw = self._get_sw()
        if step is None:
            step = self._get_step(tag)
        self.sw.add_histogram(
            tag=            tag,
            values=         values,
            global_step=    step,
            bins=           bins)

    def add_text(
            self,
            text: str,
            tag: str,
            step: Optional[int]=    None,
    ):
        if not self.sw:
            self.sw = self._get_sw()
        if step is None:
            step = self._get_step(tag)
        self.sw.add_text(
            tag=    tag,
            text_string=    text,
            global_step=    step)

    def flush(self):
        if self.sw:
            self.sw.flush()
