import numpy as np
from pypaq.lipytools.pylogger import get_pylogger
from typing import Dict, Optional, Tuple, List

from torchness.types import NPL

BATCHING_TYPES = [
    'base',         # prepares batches in order of given data
    'random',       # basic random sampling
    'random_cov']   # random sampling with full coverage of data



class BatcherException(Exception):
    pass


class Batcher:
    """
    Batcher
        takes data and prepares batches
        data for training, validation or testing is a dict: {key: np.ndarray or torch.tensor}
        key is a name of input tensor (like: 'x','feats','labels' etc.)
        batch is prepared from each key
    """

    def __init__(
            self,
            data_TR: Dict[str,NPL],
            data_VL: Optional[Dict[str,NPL]]=   None,
            data_TS: Optional[Dict[str,NPL]]=   None,
            split_VL: float=                    0.0,    # if > 0.0 and not data_VL then factor of data_TR will be put to data_VL
            split_TS: float=                    0.0,    # if > 0.0 and not data_TS then factor of data_TR will be put to data_TS
            batch_size: int=                    16,
            bs_mul: int=                        2,      # VL & TS batch_size multiplier
            batching_type: str=                 'random_cov',
            seed=                               123,
            logger=                             None):

        self.__log = logger or get_pylogger(name='Batcher')

        self.seed_counter = seed
        self.rng = np.random.default_rng(self.seed_counter)

        if batching_type not in BATCHING_TYPES:
            raise BatcherException('unknown batching_type')

        self.btype = batching_type

        self._data_keys = sorted(list(data_TR.keys()))

        if split_VL > 0 or split_TS > 0:

            if data_VL or data_TS:
                raise BatcherException('cannot split for given data_VL or data_TS')

            data_TR, data_VL, data_TS = Batcher.data_split(
                data=       data_TR,
                split_VL=   split_VL,
                split_TS=   split_TS,
                seed=       seed)

        self._data_TR = data_TR
        self._data_VL = data_VL
        self._data_TS = data_TS
        self._data_len_TR = self._data_TR[self._data_keys[0]].shape[0]

        self._batch_size = None
        self.set_batch_size(batch_size)
        self._bs_mul = bs_mul

        self._data_ixmap = []

        self._VL_batches = None
        self._TS_batches = None

        self.__log.info(f'*** Batcher *** initialized with {self._data_len_TR} samples of data in keys, batch size: {batch_size}')
        self.__log.debug('> Batcher keys:')
        for k in self._data_keys:
            self.__log.debug(f'>> {k}, shape: {self._data_TR[k].shape}, type:{type(self._data_TR[k][0])}')


    @staticmethod
    def data_split(
            data: Dict[str,NPL],
            split_VL: float,
            split_TS: float,
            seed: int):

        rng = np.random.default_rng(seed)

        keys = list(data.keys())
        d_len = len(data[keys[0]])
        indices = rng.permutation(d_len)
        nVL = int(d_len * split_VL)
        nTS = int(d_len * split_TS)
        nTR = d_len - nVL - nTS
        indices_TR = indices[:nTR]
        indices_VL = indices[nTR:nTR + nVL] if nVL else None
        indices_TS = indices[nTR + nVL:] if nTS else None

        data_TR = {}
        data_VL = {}
        data_TS = {}
        for k in keys:
            data_TR[k] = data[k][indices_TR]
            if indices_VL is not None:
                data_VL[k] = data[k][indices_VL]
            if indices_TS is not None:
                data_TS[k] = data[k][indices_TS]

        return data_TR, data_VL, data_TS


    def _extend_ixmap(self):

        if self.btype == 'base':
            self._data_ixmap += list(range(self._data_len_TR))

        if self.btype == 'random':
            self._data_ixmap += self.rng.choice(
                a=          self._data_len_TR,
                size=       self._batch_size,
                replace=    False).tolist()

        if self.btype == 'random_cov':
            new_ixmap = np.arange(self._data_len_TR)
            self.rng.shuffle(new_ixmap)
            new_ixmap = new_ixmap.tolist()
            self._data_ixmap += new_ixmap

    def set_batch_size(self, bs: int):
        if bs > self._data_len_TR:
            raise BatcherException('ERR: cannot set batch size larger than given TR data!')
        self._batch_size = bs

    def get_batch(self) -> dict:

        # set seed
        self.rng = np.random.default_rng(self.seed_counter)
        self.seed_counter += 1

        if len(self._data_ixmap) < self._batch_size: self._extend_ixmap()
        
        indexes = self._data_ixmap[:self._batch_size]
        self._data_ixmap = self._data_ixmap[self._batch_size:]
        return {k: self._data_TR[k][indexes] for k in self._data_keys}

    # splits data into batches of given size
    @staticmethod
    def __split_data(data:dict, size:int) -> List[Dict]:
        split = []
        counter = 0
        keys = list(data.keys())
        while counter*size < len(data[keys[0]]):
            split.append({k: data[k][counter*size:(counter+1)*size] for k in keys})
            counter += 1
        return split

    def get_VL_batches(self) -> List[Dict]:
        if self._VL_batches is None:
            if self._data_VL is None:
                batches = []
                self.__log.warning('Batcher asked to prepare VL batches, but no VL data was given')
            else:
                batches = Batcher.__split_data(self._data_VL, self._batch_size * self._bs_mul)
            self._VL_batches = batches
        return self._VL_batches

    def get_TS_batches(self) -> List[Dict]:
        if self._TS_batches is None:
            if self._data_TS is None:
                batches = []
                self.__log.warning('Batcher asked to prepare TS batches, but no TS data was given')
            else:
                batches = Batcher.__split_data(self._data_TS, self._batch_size * self._bs_mul)
            self._TS_batches = batches
        return self._TS_batches

    def get_data_size(self) -> Tuple[int,int,int]:
        k = self._data_keys[0]
        n_TR = self._data_TR[k].shape[0]
        n_VL = self._data_VL[k].shape[0] if self._data_VL else 0
        n_TS = self._data_TS[k].shape[0] if self._data_TS else 0
        return n_TR, n_VL, n_TS