import numpy as np
import torch
from typing import Optional, Callable, Dict, Union, Any, Iterable

ACT = Optional[type(torch.nn.Module)]       # activation type
INI = Optional[Callable]                    # initializer type

TNS = torch.Tensor                          # Tensor
DTNS = Dict[str, Union[TNS,Any]]            # dict {str: Tensor or Any}

NUM = Union[int, float, np.ndarray, TNS]    # extends pypaq NUM with Tensor
NPL = Union[Iterable[NUM], np.ndarray, TNS] # extends pypaq NPL with Tensor