from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ompr.runner import RunningWorker
from pypaq.lipytools.files import prep_folder, w_pickle, r_pickle, get_files
from pypaq.lipytools.stats import msmx

from torchness.batcher import DataBatcher, FilesBatcher, FilesBatcherMP, BATCHING_TYPES

TMP_DIR = Path(__file__).parent / '_tmp_batcher'
DATA_DIR = TMP_DIR / 'datafiles'


@pytest.fixture(autouse=True, scope='module')
def tmp_dir():
    prep_folder(TMP_DIR, flush_non_empty=True)


def test_base_init():
    data = {'input': np.random.rand(1000, 3)}
    batcher = DataBatcher(data_TR=data)  # type: ignore
    nTR, nTS = batcher.get_data_size()
    assert (nTR, nTS) == (1000, 0)
    assert batcher.keys == ['input']

    batcher = DataBatcher(data_TR=data, split_factor=0.2)  # type: ignore
    nTR, nTS = batcher.get_data_size()
    assert (nTR, nTS) == (800, 200)

    data_TS = {'input': np.random.rand(300, 3)}
    batcher = DataBatcher(data_TR=data, data_TS=data_TS)  # type: ignore
    nTR, nTS = batcher.get_data_size()
    assert (nTR, nTS) == (1000, 300)


def test_TS_batches():
    data = {'input': np.random.rand(1000, 3)}
    data_TS = {'input': np.random.rand(300, 3)}
    batcher = DataBatcher(data_TR=data, data_TS=data_TS, batch_size=15, batch_size_TS_mul=2)  # type: ignore
    batches_TS = batcher.get_TS_batches()
    assert len(batches_TS) == 10

    data = {'input': np.random.rand(1000, 3)}
    data_TS_named = {
        'test_A': {'input': np.random.rand(300, 3)},
        'test_B': {'input': np.random.rand(200, 3)},
    }
    batcher = DataBatcher(data_TR=data, data_TS=data_TS_named, batch_size=10, batch_size_TS_mul=2)  # type: ignore
    nTR, nTS = batcher.get_data_size()
    assert (nTR, nTS) == (1000, 500)
    batches_TS = batcher.get_TS_batches('test_B')
    assert len(batches_TS) == 10


def test_coverage(
        num_samples: int = 1000,
        batch_size: int = 64,
        num_batches: int = 1000):

    for btype in BATCHING_TYPES:
        print(f'\nstarts coverage tests of {btype}')

        samples = np.arange(num_samples)
        np.random.shuffle(samples)
        data = {'samples': samples}
        batcher = DataBatcher(data_TR=data, batch_size=batch_size, batching_type=btype)

        sL = []
        n_b = 0
        s_counter = {s: 0 for s in range(num_samples)}
        for _ in range(num_batches):
            sL += batcher.get_batch()['samples'].tolist()
            n_b += 1
            if len(set(sL)) == num_samples:
                print(f'got full coverage with {n_b} batches')
                for s in sL: s_counter[s] += 1
                sL = []
                n_b = 0

        print(msmx(list(s_counter.values()))['string'])
    print('*** finished coverage tests')


def test_seed():
    c_size = 1000
    b_size = 64

    samples = np.arange(c_size)
    np.random.shuffle(samples)
    data = {'samples': samples}

    batcher = DataBatcher(data, batch_size=b_size, batching_type='random')
    sA = []
    while len(sA) < 10000:
        sA += batcher.get_batch()['samples'].tolist()
        np.random.seed(len(sA))

    batcher = DataBatcher(data, batch_size=b_size, batching_type='random')
    sB = []
    while len(sB) < 10000:
        sB += batcher.get_batch()['samples'].tolist()
        np.random.seed(10000000 - len(sB))

    seed_is_fixed = all(sA[ix] == sB[ix] for ix in range(len(sA)))
    print(f'seed is fixed: {seed_is_fixed}!')
    assert seed_is_fixed


def _prepare_data_files(n_files: int = 10, nf_samples: int = 10_000):
    prep_folder(DATA_DIR, flush_non_empty=True)
    for n in range(n_files):
        data = {
            'x': np.random.rand(nf_samples, 1000),
            'y': np.arange(nf_samples) + n * nf_samples}
        w_pickle(data, f'{DATA_DIR}/f{n}.npp')


def test_FilesBatcher():
    n_files = 10
    nf_samples = 10_000
    batch_size = 1_000
    n_epochs = 5

    def chunk_builder(file: str):
        return r_pickle(file)

    print('Preparing data files for FilesBatcher ..')
    _prepare_data_files(n_files, nf_samples)

    fb = FilesBatcher(
        data_TR_chunk_fp=   get_files(str(DATA_DIR)),
        chunk_builder=      chunk_builder,
        batch_size=         batch_size)

    ys = []
    for _ in range(int(n_files * nf_samples / batch_size * n_epochs)):
        batch = fb.get_batch()
        ys += batch['y'].tolist()

    print(len(ys))
    assert len(ys) == n_files * nf_samples * n_epochs
    fb.exit()


def test_FilesBatcherMP():
    n_files = 10
    nf_samples = 10_000
    batch_size = 1_000
    n_epochs = 5

    class CB(RunningWorker):
        def process(self, file) -> Any:
            return r_pickle(file)

    print('Preparing data files for FilesBatcherMP ..')
    _prepare_data_files(n_files, nf_samples)

    fb = FilesBatcherMP(
        data_TR_chunk_fp=       get_files(str(DATA_DIR)),
        data_TS_chunk_fp=       None,
        chunk_processor_class=  CB,
        n_workers=              10,
        batch_size=             batch_size)

    ys = []
    for _ in range(int(n_files * nf_samples / batch_size * n_epochs)):
        batch = fb.get_batch()
        ys += batch['y'].tolist()

    print(len(ys))
    assert len(ys) == n_files * nf_samples * n_epochs
    fb.exit()
