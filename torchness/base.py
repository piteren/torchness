import numpy as np
import torch
from typing import Optional, Callable, Dict, Any, Sequence

ACT = Optional[type(torch.nn.Module)]   # activation type
INI = Optional[Callable]                # initializer type

ARR = np.ndarray                        # numpy array
DARR = Dict[str, ARR|Any]               # dict {str: ARR|Any}
TNS = torch.Tensor                      # torch Tensor
DTNS = Dict[str, TNS|Any]               # dict {str: TNS|Any}
DTA = Dict[str, TNS|ARR|Any]            # dict {str: TNS|ARR|Any}

NUM = int|float|ARR|TNS                 # extends pypaq NUM with TNS
NPL = Sequence[NUM]|ARR|TNS             # extends pypaq NPL with TNS


class TorchnessException(Exception):
    pass