from abc import ABC, abstractmethod
import numpy as np
from ompr.runner import OMPRunner, RunningWorker
from pypaq.lipytools.pylogger import get_pylogger, get_child
import queue
import threading
import time
import torch
from typing import Dict, Optional, Tuple, List, Union, Any

from torchness.base import ARR, TNS, NPL

BATCHING_TYPES = (
    'base',         # prepares batches (indexes) in order of given data (chunk)
    'random',       # random sampling with full coverage of data (default)
)


class BatcherException(Exception):
    pass


# splits data into batches of given size
def split_into_batches(data:Dict[str,NPL], size:int) -> List[Dict[str,NPL]]:
    split = []
    counter = 0
    keys = list(data.keys())
    while counter*size < len(data[keys[0]]):
        split.append({k: data[k][counter*size:(counter+1)*size] for k in keys})
        counter += 1
    return split


class BaseBatcher(ABC):
    """ BaseBatcher prepares batches from chunks of training (TR) and testing (TS) data.
    It is an abstract class where load_data_TR_chunk() must be implemented.
    TS input data (unnamed chunk) is a dict: {axis_name: np.ndarray or torch.Tensor}.
    TS data may be given also as a dict of named test-sets (chunks): {testset_name: {chunk}} """

    default_TS_name = '__TS__' # default name of testset when given as unnamed chunk

    def __init__(
            self,
            data_TS: Optional[Union[Dict[str,NPL], Dict[str,Dict[str,NPL]]]]=   None,
            batch_size: int=            16,
            batch_size_TS_mul: int=     2,      # VL & TS batch_size multiplier
            batching_type: str=         'random',
            seed=                       123,
            logger=                     None,
            loglevel=                   20,
    ):

        self.logger = logger or get_pylogger(
            name=   f'{self.__class__.__name__}_logger',
            level=  loglevel)

        self.seed_counter = seed
        self.rng = np.random.default_rng(self.seed_counter)

        if batching_type not in BATCHING_TYPES:
            raise BatcherException('unknown batching_type')

        self.btype = batching_type

        self._batch_size = batch_size
        self._batch_size_TS_mul = batch_size_TS_mul

        # properties below will be set when first chunk (and every next) will be loaded
        self._data_TR = {}
        self._keys = []
        self._data_TR_len = None
        self._ixmap = np.asarray([], dtype=int)
        self._ixmap_pointer = 0
        self._get_next_chunk_and_extend_ixmap()  # here first chunk is loaded

        if data_TS and type(list(data_TS.values())[0]) is not dict:
            data_TS = {self.default_TS_name: data_TS}
        self._data_TS: Dict[str,Dict[str,NPL]] = data_TS
        self._data_TS_len = sum([self._data_TS[k][self._keys[0]].shape[0] for k in self._data_TS]) if self._data_TS else 0
        self._TS_batches = {}

        self.logger.info(f'*** {self.__class__.__name__} *** initialized, batch size: {batch_size}')
        self.logger.info(f' > data_TR_len:{self._data_TR_len} - first chunk')
        if self._data_TS and list(self._data_TS.keys()) != [self.default_TS_name]:
            self.logger.info(f' > data_TS names: {list(self._data_TS.keys())}')
        self.logger.info(f' > data_TS_len: {self._data_TS_len}')
        self.logger.debug('> Batcher keys:')
        for k in self._keys:
            self.logger.debug(f'>> {k}, shape: {self._data_TR[k].shape}, type:{type(self._data_TR[k][0])}')

    @abstractmethod
    def load_data_TR_chunk(self) -> Dict[str,NPL]:
        """ should return a chunk of training data,
        it may be full epoch or just next part of it """
        pass

    def _get_next_chunk_and_extend_ixmap(self):
        """ this method is called when Batcher has not enough TR data (self._data_TR),
        to be precise: when self._ixmap is small enough """

        stime = time.time()

        chunk_next = self.load_data_TR_chunk()

        # set keys only once, with the first chunk
        if not self._keys:
            self._keys = sorted(list(chunk_next.keys()))
        chunk_next_len = chunk_next[self._keys[0]].shape[0]

        _ixmap_new = None

        if self.btype == 'base':
            _ixmap_new = np.arange(chunk_next_len)

        if self.btype == 'random':
            _ixmap_new = self.rng.choice(
                a=          chunk_next_len,
                size=       chunk_next_len,
                replace=    False)

        ### tries to concat left data with new chunk, only supported for ARR and TNS in chunks

        _ixmap_left = self._ixmap[self._ixmap_pointer:]
        _ixmap_left_size = len(_ixmap_left)

        if _ixmap_left_size:

            conc_func = None
            if type(chunk_next[self._keys[0]]) is ARR:
                conc_func = np.concatenate
            if type(chunk_next[self._keys[0]]) is TNS:
                conc_func = torch.cat

            if conc_func:
                for k in self._keys:
                    chunk_next[k] = conc_func([self._data_TR[k][_ixmap_left], chunk_next[k]])
                _ixmap_new = np.concatenate([np.arange(_ixmap_left_size), _ixmap_new+_ixmap_left_size])
            else:
                self.logger.warning(f'Batcher was unable to use left {_ixmap_left_size} samples from chunk, try using np.ndarray or torch.Tensor with TR data')

        self._ixmap = _ixmap_new
        self._ixmap_pointer = 0

        self._data_TR = chunk_next
        self._data_TR_len = self._data_TR[self._keys[0]].shape[0]

        self.logger.debug(f'> _get_next_chunk_and_extend_ixmap() took {time.time() - stime:.2f}sec')

    def get_batch(self) -> Dict[str,NPL]:

        # set seed
        self.rng = np.random.default_rng(self.seed_counter)
        self.seed_counter += 1

        if self._ixmap_pointer + self._batch_size > len(self._ixmap):
            self._get_next_chunk_and_extend_ixmap()

        indexes = self._ixmap[self._ixmap_pointer:self._ixmap_pointer+self._batch_size]
        self._ixmap_pointer += self._batch_size

        return {k: self._data_TR[k][indexes] for k in self._keys}

    def get_TS_batches(self, name:Optional[str]=None) -> List[Dict[str,NPL]]:
        """ if TS data was given as a dict of named test-sets then name (TS) has to be given,
        otherwise name=None """

        if name is None:
            if list(self._data_TS.keys()) != [self.default_TS_name]:
                raise BatcherException('ERR: TS name must be given!')
            name = self.default_TS_name

        if name not in list(self._data_TS.keys()):
            raise BatcherException('ERR: TS name unknown!')

        if name not in self._TS_batches:
            self._TS_batches[name] = split_into_batches(
                data=   self._data_TS[name],
                size=   self._batch_size * self._batch_size_TS_mul)

        return self._TS_batches[name]

    def get_data_size(self) -> Tuple[int,int]:
        """ returns current TR chunk dataset length and TS dataset length """
        return self._data_TR_len, self._data_TS_len

    def get_TS_names(self) -> Optional[List[str]]:
        if not self._data_TS:
            return None
        names = list(self._data_TS.keys())
        if names != [self.default_TS_name]:
            return names
        return None

    @property
    def keys(self) -> List[str]:
        return self._keys


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


class DataBatcher(BaseBatcher):
    """ DataBatcher prepares batches from a given training (TR) data and testing (TS) data.
    Input data is a dict: {key: np.ndarray or torch.Tensor}.
    TS data may be given as a named test-set: Dict[str, Dict[str,NPL]] """

    default_TS_name = '__TS__'

    def __init__(
            self,
            data_TR: Dict[str,NPL],
            split_factor: float=    0.0, # if > 0.0 and not data_TS then factor of data_TR will be put to data_TS
            seed=                   123,
            **kwargs,
    ):

        if split_factor > 0:

            if 'data_TS' in kwargs and kwargs['data_TS'] is not None:
                raise BatcherException('cannot split data because data_TS is given')

            data_TR, data_TS = data_split(data=data_TR, split_factor=split_factor, seed=seed)
            kwargs['data_TS'] = data_TS

        self._data_TR_chunk = data_TR

        super().__init__(seed=seed, **kwargs)

    def load_data_TR_chunk(self) -> Dict[str,NPL]:
        return {k:self._data_TR_chunk[k] for k in self._data_TR_chunk}


class FilesBatcher(BaseBatcher):
    """ FilesBatcher uses thread to load files with TR data in the background """

    def __init__(
            self,
            data_files: List[str],
            chunk_builder: callable,
            logger=         None,
            loglevel=       20,
            **kwargs,
    ):
        """
        data_files:
            list of file paths where data of TR chunks is stored
        chunk_builder(file:str):
            function that should return chunk of data Dict[str,NPL] given file path """

        self.logger = logger or get_pylogger(
            name=   f'{self.__class__.__name__}_logger',
            level=  loglevel)

        self._data_files = data_files
        if not self._data_files:
            raise BatcherException(f'data_files is empty: {data_files}')

        self._chunk_builder = chunk_builder
        self.logger.info(f'*** {self.__class__.__name__} *** initializes with {len(self._data_files)} files (TR chunks).')

        self._data_chunks = []
        self.q_to_loader = queue.Queue()

        self.loader_thread = threading.Thread(target=self._loader_loop)
        self.loader_thread.start()
        self.q_to_loader.put('load') # put the first task

        super().__init__(logger=self.logger, **kwargs)
        
    def _loader_loop(self):
        
        self.logger.debug('loader started loop')

        while True:

            stime = time.time()
            msg = self.q_to_loader.get()
            self.logger.debug(f'> loader thread waited {time.time() - stime:.2f}sec for a new task (msg)')

            if msg == 'load':

                stime = time.time()

                file = self._data_files.pop(0)
                self._data_files.append(file)

                self.logger.debug(f'>> loader starts loading file: {file} ..')
                _data = self._chunk_builder(file=file)
                self._data_chunks.append(_data)
                self.logger.debug(f'>> loader added chunk of data from file: {file}, thread took {time.time()-stime:.2f}sec')

            if msg == 'exit':
                break

    def load_data_TR_chunk(self) -> Dict[str,NPL]:
        stime = time.time()
        while True:
            if self._data_chunks:
                _waited = time.time() - stime
                data = self._data_chunks.pop(0)
                self.q_to_loader.put('load') # put next task immediately
                self.logger.debug(f'> load_data_TR_chunk() waited {_waited:.2f}sec '
                                  f'for a new data chunk, TOT:{time.time() - stime:.2f}sec')
                return data
            time.sleep(0.1)

    def exit(self):
        self.q_to_loader.put('exit')
        self.loader_thread.join()


class FilesBatcherMP(BaseBatcher):
    """ FilesBatcherMP uses OMPR (subprocesses) to load files with TR data in the background """

    def __init__(
            self,
            data_files: List[str],
            chunk_builder: callable,
            n_workers: int= 5,
            logger=         None,
            loglevel=       20,
            **kwargs,
    ):
        """
        data_files:
            list of file paths where data of TR chunks is stored
        chunk_builder(file:str):
            function that should return chunk of data Dict[str,NPL] given file path
        n_workers:
            max number of parallel MP workers that will be put into the chunk loading task,
            when average time needed by a worker to load single chunk is greater
            than time of running this chunk with a NN, number of workers > 1 """

        self.logger = logger or get_pylogger(
            name=   f'{self.__class__.__name__}_logger',
            level=  loglevel)

        self._data_files = data_files
        if not self._data_files:
            raise BatcherException(f'data_files is empty: {data_files}')

        self._chunk_builder = chunk_builder
        self.logger.info(f'*** {self.__class__.__name__} *** initializes with {len(self._data_files)} files (TR chunks).')

        class ChunkBuilder(RunningWorker):
            def process(self, **kwargs) -> Any:
                return chunk_builder(**kwargs)

        logger_child = get_child(logger=self.logger, name=f'{self.logger.name}_child', change_level=10)
        self.ompr = OMPRunner(
            rww_class=          ChunkBuilder,
            devices=            [None] * n_workers,
            ordered_results=    False,
            rerun_crashed=      False,
            logger=             logger_child)

        for _ in range(n_workers):
            self._put_next_task_to_ompr()

        super().__init__(logger=self.logger, **kwargs)

    def _put_next_task_to_ompr(self):
        file = self._data_files.pop(0)
        self._data_files.append(file)
        self.ompr.process({'file':file})

    def load_data_TR_chunk(self) -> Dict[str,NPL]:
        stime = time.time()
        data = self.ompr.get_result()
        self.logger.debug(f'> load_data_TR_chunk() waited {time.time() - stime:.2f}sec for a new data chunk')
        self._put_next_task_to_ompr() # put next task immediately
        return data

    def exit(self):
        self.ompr.exit()