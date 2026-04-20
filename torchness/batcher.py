from abc import ABC, abstractmethod
import numpy as np

from ompr.runner import OMPRunner, RunningWorker
from pypaq.lipytools.pylogger import get_pylogger, get_child
import queue
import threading
import time
from typing import Callable

from torchness.base import DATNS, cat_arrays, copy_array

BATCHING_TYPES = (
    'base',     # prepares batches (indexes) in order of given data (chunk)
    'random',   # random sampling with full coverage of data (default)
)


class BatcherException(Exception):
    pass


def split_into_batches(data:DATNS, size:int) -> list[DATNS]:
    """splits data into batches of given size"""
    split = []
    counter = 0
    keys = list(data.keys())
    while counter*size < len(data[keys[0]]):
        split.append({k: data[k][counter*size:(counter+1)*size] for k in keys})
        counter += 1
    return split


def data_split(
        data: DATNS,
        split_factor: float,    # factor of data separated into second set
        seed: int = 123,
) -> tuple[DATNS, DATNS]:
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


class BaseBatcher(ABC):
    """ BaseBatcher prepares batches from chunks of training (TR) and testing (TS) data.
    It is an abstract class where load_data_TR_chunk() must be implemented.
    TS input data (unnamed chunk) is a dict: {axis_name: np.ndarray or torch.Tensor}.
    TS data may be given also as a dict of named test-sets (chunks): {testset_name: {chunk}} """

    default_TS_name = '__TS__' # default name of testset when given as (unnamed) DATNS

    def __init__(
            self,
            data_TS: DATNS | dict[str, DATNS] | None = None,
            batch_size: int = 16,
            batch_size_TS_mul: int = 2,  # TS batch_size multiplier
            batching_type: str = 'random',
            seed: int = 123,
            timing_report: bool = False,
            logger = None,
            loglevel :int = 20,
    ):
        """device: if given, moves TR and TS data to device"""
        if batching_type not in BATCHING_TYPES:
            raise BatcherException('unknown batching_type')

        self.logger = logger or get_pylogger(
            name=   f'{self.__class__.__name__}_logger',
            level=  loglevel)

        self._timing = {
            'load_chunk': [],
            'extend_ixmap': [],
        } if timing_report else None

        self.seed_counter = seed
        self.rng = np.random.default_rng(self.seed_counter)

        self.btype = batching_type

        self._batch_size = batch_size
        self._batch_size_TS_mul = batch_size_TS_mul

        # attr below will be set when the first chunk will be loaded
        self._data_TR = {}
        self._keys = []
        self._data_TR_len = None
        self._ixmap = np.asarray([], dtype=int)
        self._ixmap_pointer = 0
        self._get_next_chunk_and_extend_ixmap()  # load first chunk

        if data_TS and type(list(data_TS.values())[0]) is not dict:
            data_TS = {self.default_TS_name: data_TS}
        self._data_TS: dict[str, DATNS] = data_TS # type: ignore
        self._data_TS_len = sum([
            self._data_TS[k][self._keys[0]].shape[0] for k in self._data_TS]
        ) if self._data_TS else 0
        self._TS_batches = {}

        self.logger.info(f'*** {self.__class__.__name__} *** initialized, batch size: {batch_size}')
        self.logger.info(f'> data_TR_len: {self._data_TR_len} - loaded (first?) chunk')
        if self._data_TS and list(self._data_TS.keys()) != [self.default_TS_name]:
            self.logger.info(f'> data_TS names: {list(self._data_TS.keys())}')
        self.logger.info(f'> data_TS_len: {self._data_TS_len}')
        self.logger.debug('> Batcher (batch) keys:')
        for k in self._keys:
            self.logger.debug(f'>> {k}, shape: {self._data_TR[k].shape}, type:{type(self._data_TR[k][0])}')

    @abstractmethod
    def load_data_TR_chunk(self) -> DATNS:
        """ should return a chunk of training data,
        it may be a full epoch or just a next part of it """
        pass

    def _get_next_chunk_and_extend_ixmap(self):
        """ this method is called when Batcher has not enough TR data (self._data_TR),
        precisely: when self._ixmap is small enough """

        stime = time.time()

        chunk_next = self.load_data_TR_chunk()

        td = time.time() - stime
        self.logger.debug(f'> load_data_TR_chunk() waited {td:.2f}sec for a new data chunk')
        if self._timing:
            self._timing['load_chunk'].append(td)

        stime = time.time()

        # set keys only once, with the first chunk
        if not self._keys:
            self._keys = sorted(list(chunk_next.keys()))
        chunk_next_len = len(chunk_next[self._keys[0]])

        _ixmap_new = np.arange(chunk_next_len) # base
        if self.btype == 'random':
            self.rng.shuffle(_ixmap_new)

        # concat left data with new chunk
        _ixmap_left = self._ixmap[self._ixmap_pointer:]
        _ixmap_left_size = len(_ixmap_left)
        if _ixmap_left_size:
            for k in self._keys:
                chunk_next[k] = cat_arrays([self._data_TR[k][_ixmap_left], chunk_next[k]])
            _ixmap_new = np.concatenate([np.arange(_ixmap_left_size), _ixmap_new+_ixmap_left_size])

        self._ixmap = _ixmap_new
        self._ixmap_pointer = 0

        self._data_TR = chunk_next
        self._data_TR_len = len(self._data_TR[self._keys[0]])

        td = time.time() - stime
        self.logger.debug(f'> _get_next_chunk_and_extend_ixmap() took {td:.2f}sec')
        if self._timing:
            self._timing['extend_ixmap'].append(td)

    def get_batch(self) -> DATNS:

        # set seed
        self.rng = np.random.default_rng(self.seed_counter)
        self.seed_counter += 1

        if self._ixmap_pointer + self._batch_size > len(self._ixmap):
            self._get_next_chunk_and_extend_ixmap()

        indexes = self._ixmap[self._ixmap_pointer:self._ixmap_pointer+self._batch_size]
        self._ixmap_pointer += self._batch_size

        return {k: self._data_TR[k][indexes] for k in self._keys}

    def get_TS_batches(self, name:str|None=None) -> list[DATNS]:
        """ if TS data was given as a dict of named test-sets then name (TS) has to be given,
        otherwise name=None """

        if not self._data_TS:
            return []

        if name is None:
            if list(self._data_TS.keys()) != [self.default_TS_name]:
                raise BatcherException('ERR: TS name must be given!')
            name = self.default_TS_name

        if name not in list(self._data_TS.keys()):
            raise BatcherException('ERR: TS name unknown!')

        if name not in self._TS_batches:
            data = self._data_TS[name]
            self._TS_batches[name] = split_into_batches(
                data=   data,
                size=   self._batch_size * self._batch_size_TS_mul)

        return self._TS_batches[name]

    def get_data_size(self) -> tuple[int,int]:
        """ returns current TR chunk dataset length and TS dataset length """
        return self._data_TR_len, self._data_TS_len

    def get_TS_names(self) -> list[str]|None:
        if not self._data_TS:
            return None
        return list(self._data_TS.keys())

    @property
    def keys(self) -> list[str]:
        return self._keys

    def exit(self):
        self.logger.info(f'{self.__class__.__name__} exits ..')
        if self._timing:
            n_loads = len(self._timing['load_chunk'])
            if n_loads:
                load_t = sum(self._timing['load_chunk']) / n_loads
                self.logger.info(f'> loads ({n_loads}), avg: {load_t:.2f}sec')
            n_extends = len(self._timing['extend_ixmap'])
            if n_extends:
                ext_t = sum(self._timing['extend_ixmap']) / n_extends
                self.logger.info(f'> extensions ({n_extends}), avg: {ext_t:.2f}sec')


class DataBatcher(BaseBatcher):
    """ DataBatcher prepares batches from a given training (TR) data and testing (TS) data.
    Input data is a dict: {key: np.ndarray or torch.Tensor}.
    TS data may be given as a named test-set: Dict[str, DATNS] """

    default_TS_name = '__TS__'

    def __init__(
            self,
            data_TR: DATNS,
            split_factor: float=    0.0, # if > 0.0 and not data_TS then factor of data_TR will be put to data_TS
            seed: int = 123,
            **kwargs,
    ):

        if split_factor > 0:

            if 'data_TS' in kwargs and kwargs['data_TS'] is not None:
                raise BatcherException('cannot split data because data_TS is given')

            data_TR, data_TS = data_split(data=data_TR, split_factor=split_factor, seed=seed)
            kwargs['data_TS'] = data_TS

        self._data_TR_chunk = data_TR

        super().__init__(seed=seed, **kwargs)

    def load_data_TR_chunk(self) -> DATNS:
        return {k:self._data_TR_chunk[k] for k in self._data_TR_chunk}


class FilesBatcher(BaseBatcher):
    """ FilesBatcher uses threads to load data files in the background.
    It is designed for cases, where data does is saved in files,
    fast file read is crucial for the training speed
    and data does not need any expensive processing before loading into the NN. """

    def __init__(
            self,
            data_files_fp: list[str],
            chunk_builder: Callable,
            logger = None,
            loglevel :int = 20,
            **kwargs,
    ):
        """
        data_TR_chunk_fp:
            list of file paths where data of TR chunks is stored
        chunk_builder(file:str):
            function that should return chunk of data DATNS given file path """

        self.logger = logger or get_pylogger(
            name=   f'{self.__class__.__name__}_logger',
            level=  loglevel)

        self._data_files_fp = data_files_fp

        self._chunk_builder = chunk_builder
        self.logger.info(f'*** {self.__class__.__name__} *** initializes with {len(self._data_files_fp)} files')

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

                file = self._data_files_fp.pop(0)
                self._data_files_fp.append(file)

                self.logger.debug(f'>> loader starts loading file: {file} ..')
                _data = self._chunk_builder(file=file)
                self._data_chunks.append(_data)
                self.logger.debug(f'>> loader added chunk of data from file: {file}, thread took {time.time()-stime:.2f}sec')

            if msg == 'exit':
                break

    def load_data_TR_chunk(self) -> DATNS:
        while True:
            if self._data_chunks:
                data = self._data_chunks.pop(0)
                self.q_to_loader.put('load') # put next task immediately
                return data
            time.sleep(0.1)

    def exit(self):
        super().exit()
        self.q_to_loader.put('exit')
        self.loader_thread.join()


class FilesBatcherMP(BaseBatcher):
    """ FilesBatcherMP uses subprocesses (OMPR) to load data files in the background.
    It is designed for cases, where data saved in files needs further processing before loading into NN,
    and read operations are less critical than processing. """

    def __init__(
            self,
            data_TR_chunk_fp: list[str],
            data_TS_chunk_fp: str | dict[str,str] | None,
            chunk_processor_class: type[RunningWorker],
            rww_init_kwargs: dict | None = None,
            n_workers: int = 5,
            ordered_results: bool = True,
            raise_rww_exception: bool = False,
            logger = None,
            loglevel: int = 20,
            **kwargs,
    ):
        """
        data_TR_chunk_fp:
            list of file paths with TR data chunks
        data_TS_chunk_fp:
            file path with TS data chunks
        chunk_processor_class:
            is a class of type RunningWorker, where process accepts file_fp(str)
            and returns a NN ready chunk of data DATNS
        rww_init_kwargs:
            chunk_processor_class __init__ kwargs
        n_workers:
            max number of parallel MP workers that will be put into the chunk loading task,
            when average time needed by a worker to load and prepare a single chunk is greater
            than time of running (training) this chunk with a NN, number of workers > 1
        ordered_results:
            allows for reproducibility of results,
            REMEMBER to keep order of data_TR files """

        self.logger = logger or get_pylogger(
            name=   f'{self.__class__.__name__}_logger',
            level=  loglevel)

        n_test_files = 0 if not data_TS_chunk_fp else (1 if type(data_TS_chunk_fp) is str else len(data_TS_chunk_fp))
        self._data_TR_chunk_fp = data_TR_chunk_fp
        self.logger.info(f'*** {self.__class__.__name__} *** initializes with {len(self._data_TR_chunk_fp)} TR files, '
                         f'{n_test_files} TS files, n_workers:{n_workers}')
        self.static_data: bool | list = n_workers >= len(self._data_TR_chunk_fp)
        if self.static_data:
            if n_workers > len(self._data_TR_chunk_fp):
                n_workers = len(self._data_TR_chunk_fp)
                self.logger.info(f'> reduced n_workers to {n_workers} for static data case')

        self.ompr = OMPRunner(
            rww_class=              chunk_processor_class,
            rww_init_kwargs=        rww_init_kwargs,
            devices=                [None] * n_workers,
            ordered_results=        ordered_results,
            rerun_crashed=          False,
            raise_rww_exception=    raise_rww_exception,
            logger=                 get_child(logger=self.logger, change_level=10))

        for _ in range(n_workers):
            self._put_next_task_to_ompr()
        if self.static_data:
            self.static_data = self.ompr.get_all_results()
            self.ompr.exit()

        data_TS = None
        if data_TS_chunk_fp:
            self.logger.info(f'processing data_TS_chunk ..')
            cb_kwargs = {}
            if rww_init_kwargs:
                cb_kwargs.update(rww_init_kwargs)
            cb = chunk_processor_class(**cb_kwargs)
            if type(data_TS_chunk_fp) is str:
                data_TS = cb.process(file=data_TS_chunk_fp)
                self.logger.info(f'> loaded and processed data_TS_chunk from {data_TS_chunk_fp}')
            else:
                data_TS = {}
                for k,fp in data_TS_chunk_fp.items():
                    data_TS[k] = cb.process(file=fp)
                    self.logger.info(f'> loaded and processed data_TS_chunk {k} from {fp}')

        super().__init__(data_TS=data_TS, logger=self.logger, **kwargs)

    def _put_next_task_to_ompr(self):
        file = self._data_TR_chunk_fp.pop(0)
        self._data_TR_chunk_fp.append(file)
        self.ompr.process({'file':file})

    def load_data_TR_chunk(self) -> DATNS:
        if self.static_data:
            data = self.static_data.pop(0)
            data_copy = {k: copy_array(data[k]) for k in data} # keep clean copy
            self.static_data.append(data_copy)
        else:
            data = self.ompr.get_result()
            self._put_next_task_to_ompr()  # put next task immediately
        return data # type: ignore

    def exit(self):
        super().exit()
        if not self.static_data:
            self.ompr.exit()