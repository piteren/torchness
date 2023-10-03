import numpy as np
from pypaq.pytypes import NPL
from typing import Optional, List, Dict


# processes zeroes arrays accumulated in intervals
class ZeroesProcessor:

    def __init__(
            self,
            intervals: tuple=   (50,500,5000),
            tag_pfx=            'nane',     # prefix of tag in TB, (Not Activated NEurons)
            tbwr: Optional=     None):      # if given will put summaries to TB with intervals frequencies
        self.intervals = intervals
        self.zsL = {k: [] for k in self.intervals}
        self.single = []
        self.tag_pfx = tag_pfx
        self.tbwr = tbwr
        self.step = 0

    # takes next zeroes array and processes
    def process(self, zs:List[NPL], step:Optional[int]=None) -> Dict[int,float]:

        if step is None:
            step = self.step

        zs = np.concatenate(zs)
        self.single.append(float(np.mean(zs)))

        iv_NANE = {}
        if len(self.single) == self.intervals[0]:
            iv_NANE[1] = np.mean(self.single)
            self.single = []

        for k in self.zsL:
            self.zsL[k].append(zs)
            if len(self.zsL[k]) == k:
                stacked = np.stack(self.zsL[k], axis=0)     # stack arrays
                mean = np.mean(stacked, axis=0)             # mean along 0 axis (averages non activated over k)
                clipped = np.where(mean==1,1,0)             # where average over k is 1 leave 1, else 0
                iv_NANE[k] = float(np.mean(clipped))        # factor of neurons not activated (1) over k
                self.zsL[k] = []                            # reset

        if self.tbwr:
            for k in iv_NANE:
                self.tbwr.add(
                    value=  iv_NANE[k],
                    tag=    f'{self.tag_pfx}/nane_{k}',
                    step=   step)

        self.step += 1

        return iv_NANE