from pypaq.pytypes import ARR
import torch
from typing import Callable, Sequence, Type

ACT = Type[torch.nn.Module] | None  # activation type
INI = Callable | None               # initializer type

TNS = torch.Tensor                  # torch Tensor
ATNS = ARR | TNS                    # numpy array or torch Tensor
NUM = int | float | ATNS            # extends pypaq NUM with TNS
NPL = Sequence[NUM] | ATNS          # extends pypaq NPL with TNS
DTNS = dict[str, TNS]
DATNS = dict[str, ATNS]


class TorchnessException(Exception):
    pass


def cat_arrays(arrays:list[ATNS]) -> ATNS:
    return np.concatenate(arrays) if type(arrays[0]) is ARR else torch.cat(arrays) # type: ignore


def copy_array(a:ATNS) -> ATNS:
    return np.copy(a) if type(a) is ARR else torch.clone(a) # type: ignore