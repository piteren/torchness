import numpy as np
import torch
from typing import Optional, Callable, Dict, Union, Any, Sequence


ACT = Optional[type(torch.nn.Module)]       # activation type
INI = Optional[Callable]                    # initializer type

ARR = np.ndarray                            # numpy array
TNS = torch.Tensor                          # torch Tensor
DTNS = Dict[str, Union[TNS,Any]]            # dict {str: TNS or Any}

NUM = Union[int, float, ARR, TNS]           # extends pypaq NUM with TNS
NPL = Union[Sequence[NUM], ARR, TNS]        # extends pypaq NPL with TNS


class TorchnessException(Exception):
    pass