import torch
from typing import Optional, Dict, List, Union

from torchness.base import TNS, NUM


class ZeroesProcessor:
    """ ZeroesProcessor
    processes (analyzes) zeroes arrays
    usually accumulated by NN in intervals of FWD/BWD """

    def __init__(
            self,
            intervals: tuple=   (50,500,5000),
            tag_pfx=            'nane',     # prefix of tag in TB, (Not Activated NEurons)
            tbwr: Optional=     None):      # if given will put summaries to TB with intervals frequencies
        self.intervals = intervals
        self.zsDL: Dict[int,List[TNS]] = {k: [] for k in self.intervals}
        self.single: List[TNS] = []
        self.tag_pfx = tag_pfx
        self.tbwr = tbwr
        self.step = 0

    @staticmethod
    def _extract_TNS_from(l:List) -> List[TNS]:
        """ extract TNS from a (nested) list of TNS """
        tL = []
        for e in l:
            if type(e) is list: tL += ZeroesProcessor._extract_TNS_from(e)
            else:               tL.append(e)
        return tL

    def process(
            self,
            zeroes: Union[TNS,List],
            step: Optional[int]=None
    ) -> Dict[int,NUM]:
        """ processes next zeroes
        returned dict may be empty if no interval passed
        zeroes may be given as Tensor or (nested) list of Tensor """

        iv_nane = {}

        if step is None:
            step = self.step

        with torch.no_grad():

            if type(zeroes) is list:
                zeroes = ZeroesProcessor._extract_TNS_from(zeroes)
                if not zeroes:
                    return iv_nane
                zeroes = torch.cat(zeroes)

            self.single.append(torch.mean(zeroes, dtype=torch.float32))

            if len(self.single) == self.intervals[0]:
                iv_nane[1] = torch.Tensor(self.single).mean()
                self.single = []

            for k in self.zsDL:
                self.zsDL[k].append(zeroes)
                if len(self.zsDL[k]) == k:
                    stacked = torch.stack(self.zsDL[k], dim=0)
                    mean = torch.mean(stacked, dim=0, dtype=torch.float32)  # mean along 0 axis (averages non activated over k)
                    clipped = (mean==1).to(torch.float32)                   # where average over k is 1 leave 1, else 0
                    iv_nane[k] = clipped.mean()                             # factor of neurons not activated (1) over k
                    self.zsDL[k] = []                                       # reset

        if self.tbwr:
            for k in iv_nane:
                self.tbwr.add(
                    value=  iv_nane[k],
                    tag=    f'{self.tag_pfx}/nane_{k}',
                    step=   step)

        self.step += 1

        return iv_nane