import numpy as np
from pypaq.lipytools.pylogger import get_pylogger
from typing import Dict, Optional, Tuple, List, Union

from torchness.types import NPL

BATCHING_TYPES = (
    'base',         # prepares batches in order of given data
    'random',       # basic random sampling
    'random_cov',   # random sampling with full coverage of data (default)
)


class BatcherException(Exception):
    pass


def data_split(
        data: Dict[str,NPL],
        split_factor: float,    # factor of data separated into second set
        seed: int=      123,
) -> Tuple[Dict[str,NPL], Dict[str,NPL]]:
    """ splits given data into two sets """

    rng = np.random.default_rng(seed)

    keys = list(data.keys())
    d_len = len(data[keys[0]])
    indices = rng.permutation(d_len)

    splitB_len = int(d_len * split_factor)
    splitA_len = d_len - splitB_len

    indicesA = indices[:splitA_len]
    indicesB = indices[splitA_len:]

    dataA = {}
    dataB = {}
    for k in keys:
        dataA[k] = data[k][indicesA]
        dataB[k] = data[k][indicesB]

    return dataA, dataB


class Batcher:
    """ Batcher prepares batches from given training (TR) and testing (TS) data.
    Input data is a dict: {key: np.ndarray or torch.Tensor}.
    TS data may be given as a named test-set: Dict[str, Dict[str,NPL]] """

    default_TS_name = '__TS__'

    def __init__(
            self,
            data_TR: Dict[str,NPL],
            data_TS: Optional[Union[Dict[str,NPL], Dict[str,Dict[str,NPL]]]]=   None,
            split_factor: float=        0.0,    # if > 0.0 and not data_TS then factor of data_TR will be put to data_TS
            batch_size: int=            16,
            batch_size_TS_mul: int=     2,      # VL & TS batch_size multiplier
            batching_type: str=         'random_cov',
            seed=                       123,
            logger=                     None,
            loglevel=                   20,
    ):

        self.__log = logger or get_pylogger(
            name=   f'{self.__class__.__name__}_logger',
            level=  loglevel)

        self.seed_counter = seed
        self.rng = np.random.default_rng(self.seed_counter)

        if batching_type not in BATCHING_TYPES:
            raise BatcherException('unknown batching_type')

        self.btype = batching_type

        self._keys = sorted(list(data_TR.keys()))

        if split_factor > 0:

            if data_TS:
                raise BatcherException('cannot split data because data_TS is given')

            data_TR, data_TS = data_split(data=data_TR, split_factor=split_factor, seed=seed)

        self._data_TR = data_TR
        self._data_TR_len = self._data_TR[self._keys[0]].shape[0]

        if data_TS and type(list(data_TS.values())[0]) is not dict:
            data_TS = {Batcher.default_TS_name: data_TS}
        self._data_TS: Dict[str,Dict[str,NPL]] = data_TS
        self._data_TS_len = sum([self._data_TS[k][self._keys[0]].shape[0] for k in self._data_TS]) if self._data_TS else 0
        self._TS_batches = {}

        self._batch_size = 0
        self.set_batch_size(batch_size)
        self._batch_size_TS_mul = batch_size_TS_mul

        self._data_ixmap = np.asarray([], dtype=int)
        self._ixmap_pointer = 0

        self.__log.info(f'*** Batcher *** initialized, batch size: {batch_size}')
        self.__log.info(f' > data_TR_len: {self._data_TR_len}')
        if self._data_TS and list(self._data_TS.keys()) != [Batcher.default_TS_name]:
            self.__log.info(f' > data_TS names: {list(self._data_TS.keys())}')
        self.__log.info(f' > data_TS_len: {self._data_TS_len}')
        self.__log.debug('> Batcher keys:')
        for k in self._keys:
            self.__log.debug(f'>> {k}, shape: {self._data_TR[k].shape}, type:{type(self._data_TR[k][0])}')


    def _extend_ixmap(self):

        _new_ixmap = None

        if self.btype == 'base':
            _new_ixmap = np.arange(self._data_TR_len)

        if self.btype == 'random':
            _new_ixmap = self.rng.choice(
                a=          self._data_TR_len,
                size=       self._batch_size,
                replace=    False)

        if self.btype == 'random_cov':
            _new_ixmap = np.arange(self._data_TR_len)
            self.rng.shuffle(_new_ixmap)

        self._data_ixmap = np.concatenate([self._data_ixmap[self._ixmap_pointer:], _new_ixmap])
        self._ixmap_pointer = 0

    def set_batch_size(self, bs:int):
        if bs > self._data_TR_len:
            raise BatcherException('ERR: cannot set batch size > TR data!')
        self._batch_size = bs
        self._TS_batches = {}

    def get_batch(self) -> dict:

        # set seed
        self.rng = np.random.default_rng(self.seed_counter)
        self.seed_counter += 1

        if self._ixmap_pointer + self._batch_size > len(self._data_ixmap):
            self._extend_ixmap()
        
        indexes = self._data_ixmap[self._ixmap_pointer:self._ixmap_pointer+self._batch_size]
        self._ixmap_pointer += self._batch_size

        return {k: self._data_TR[k][indexes] for k in self._keys}

    # splits data into batches of given size
    @staticmethod
    def __split_into_batches(data:dict, size:int) -> List[Dict]:
        split = []
        counter = 0
        keys = list(data.keys())
        while counter*size < len(data[keys[0]]):
            split.append({k: data[k][counter*size:(counter+1)*size] for k in keys})
            counter += 1
        return split

    def get_TS_batches(self, name:Optional[str]=None) -> List[Dict]:
        """ if TS data was given as a named test-set then name has to be filled,
        otherwise name=None """

        if name is None:
            if list(self._data_TS.keys()) != [Batcher.default_TS_name]:
                raise BatcherException('ERR: TS name must be given!')
            name = Batcher.default_TS_name

        if name not in list(self._data_TS.keys()):
            raise BatcherException('ERR: TS name unknown!')

        if name not in self._TS_batches:
            self._TS_batches[name] = Batcher.__split_into_batches(
                data=   self._data_TS[name],
                size=   self._batch_size * self._batch_size_TS_mul)

        return self._TS_batches[name]

    def get_data_size(self) -> Tuple[int,int]:
        return self._data_TR_len, self._data_TS_len

    def get_TS_names(self) -> Optional[List[str]]:
        if not self._data_TS:
            return None
        names = list(self._data_TS.keys())
        if names != [Batcher.default_TS_name]:
            return names
        return None

    @property
    def keys(self) -> List[str]:
        return self._keys