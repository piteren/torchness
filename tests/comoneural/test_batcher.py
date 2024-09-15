import numpy as np
from pypaq.lipytools.files import prep_folder, w_pickle, list_dir, r_pickle
import unittest

from tests.envy import flush_tmp_dir

from torchness.comoneural.batcher import DataBatcher, FilesBatcher, BATCHING_TYPES
from pypaq.lipytools.stats import msmx

BATCHER_DATA_DIR = f'{flush_tmp_dir()}/comoneural/batcher/datafiles'


class TestDataBatcher(unittest.TestCase):

    def test_base_init(self):

        data = {'input': np.random.rand(1000,3)}
        batcher = DataBatcher(data_TR=data)
        nTR, nTS = batcher.get_data_size()
        self.assertTrue((nTR,nTS)==(1000,0))
        self.assertTrue(batcher.keys==['input'])

        batcher = DataBatcher(data_TR=data, split_factor=0.2)
        nTR, nTS = batcher.get_data_size()
        self.assertTrue((nTR, nTS) == (800,200))

        data_TS = {'input': np.random.rand(300,3)}
        batcher = DataBatcher(data_TR=data, data_TS=data_TS)
        nTR, nTS = batcher.get_data_size()
        self.assertTrue((nTR, nTS) == (1000,300))

    def test_TS_batches(self):

        data = {'input': np.random.rand(1000,3)}
        data_TS = {'input': np.random.rand(300,3)}
        batcher = DataBatcher(data_TR=data, data_TS=data_TS, batch_size=15, batch_size_TS_mul=2)
        batches_TS =batcher.get_TS_batches()
        self.assertTrue(len(batches_TS)==10)

        data = {'input': np.random.rand(1000,3)}
        data_TS = {
            'test_A': {'input': np.random.rand(300,3)},
            'test_B': {'input': np.random.rand(200,3)},
        }
        batcher = DataBatcher(data_TR=data, data_TS=data_TS, batch_size=10, batch_size_TS_mul=2)
        nTR, nTS = batcher.get_data_size()
        self.assertTrue((nTR, nTS) == (1000,500))
        batches_TS = batcher.get_TS_batches('test_B')
        self.assertTrue(len(batches_TS) == 10)


    def test_coverage(
            self,
            num_samples=    1000,
            batch_size=     64,
            num_batches=    1000):

        for btype in BATCHING_TYPES:
            print(f'\nstarts coverage tests of {btype}')

            samples = np.arange(num_samples)
            np.random.shuffle(samples)

            data = {'samples':samples}

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
        print(f' *** finished coverage tests')

    # test for Batcher reproducibility with seed
    def test_seed(self):

        c_size = 1000
        b_size = 64

        samples = np.arange(c_size)
        np.random.shuffle(samples)

        data = {'samples':samples}

        for batching_type in ['random','random_cov']:

            batcher = DataBatcher(data, batch_size=b_size, batching_type=batching_type)
            sA = []
            while len(sA) < 10000:
                sA += batcher.get_batch()['samples'].tolist()
                np.random.seed(len(sA))

            batcher = DataBatcher(data, batch_size=b_size, batching_type=batching_type)
            sB = []
            while len(sB) < 10000:
                sB += batcher.get_batch()['samples'].tolist()
                np.random.seed(10000000-len(sB))

            seed_is_fixed = True
            for ix in range(len(sA)):
                if sA[ix] != sB[ix]:
                    seed_is_fixed = False

            print(f'seed is fixed for {batching_type}: {seed_is_fixed}!')
            self.assertTrue(seed_is_fixed)


class TestFilesBatcher(unittest.TestCase):

    def test_base(self):

        n_files = 6
        nf_samples = 50000
        batch_size = 1000

        def chunk_builder(file:str):
            return r_pickle(file)

        print('Preparing data files for FilesBatcher ..')
        prep_folder(BATCHER_DATA_DIR, flush_non_empty=True)
        for n in range(n_files):
            data = {
                'x':    np.random.rand(nf_samples,1000),
                'y':    np.arange(nf_samples) + n*nf_samples,
            }
            w_pickle(data, f'{BATCHER_DATA_DIR}/f{n}.npp')

        fb = FilesBatcher(
            data_files=     [f'{BATCHER_DATA_DIR}/{f}' for f in list_dir(BATCHER_DATA_DIR)['files']],
            chunk_builder=  chunk_builder,
            batch_size=     batch_size,
            loglevel=       10)

        ys = []
        for _ in range(int(n_files*nf_samples/batch_size)):
            batch = fb.get_batch()
            ys += batch['y'].tolist()

        print(len(ys), len(set(ys)))
        self.assertTrue(len(set(ys)) == n_files * nf_samples)

        fb.exit()