import torch
import unittest

from torchness.base import TorchnessException
from torchness.encoders import LayBlockDRT, EncDRT, LayBlockCNN, EncCNN, LayBlockTNS, EncTNS


class TestEncoders(unittest.TestCase):

    def test_LayBlockDRT_init(self):
        lay_drt = LayBlockDRT(10, in_lay_norm=False)
        print(lay_drt)

    def test_LayBlockDRT_init_more(self):
        lay_drt = LayBlockDRT(
            in_width=           10,
            do_scaled_dns=      True,
            interlay_dropout=   0.1,
            lay_dropout=        0.1,
            res_dropout=        0.1)
        print(lay_drt)

    def test_LayBlockDRT_base(self):

        in_width = 10
        inp = torch.rand(in_width) - 0.5
        print(inp, inp.dtype)

        lay_drt = LayBlockDRT(in_width)
        print(lay_drt)
        out = lay_drt(inp)
        print(out)
        self.assertTrue(out['out'].shape[-1] == in_width)
        self.assertTrue(out['zeroes'].shape[-1] == in_width)

    def test_LayBlockDRT_kwargs(self):

        in_width = 10
        inp = torch.rand(in_width) - 0.5

        lay_drt = LayBlockDRT(
            in_width=           in_width,
            do_scaled_dns=      True,
            dns_scale=          4,
            interlay_dropout=   0.1,
            lay_dropout=        0.1,
            res_dropout=        0.1)
        print(lay_drt)
        out = lay_drt(inp)
        print(out)
        self.assertTrue(out['out'].shape[-1] == in_width)
        self.assertTrue(out['zeroes'].shape[-1] == in_width * 4)

    def test_LayBlockDRT_device(self):

        in_width = 10
        inp = torch.rand(in_width) - 0.5

        lay_drt = LayBlockDRT(in_width)

        dev = torch.device('cuda')
        lay_drt = lay_drt.to(dev)
        inp = inp.to(dev)
        out = lay_drt(inp)
        print(out['out'].device)
        print(out)

        lay_drt = LayBlockDRT(in_width)
        self.assertRaises(RuntimeError, lay_drt, inp)  # devices mismatch

    def test_LayBlockDRT_double(self):

        in_width = 10
        inp = torch.rand(in_width) - 0.5

        lay_drt = LayBlockDRT(
            in_width=       in_width,
            do_scaled_dns=  True,
            lay_dropout=    0.1,
            res_dropout=    0.1)
        print(lay_drt)
        out = lay_drt(inp)
        print(out['out'].dtype)
        print(out)

    def test_EncDRT_init(self):
        enc_drt = EncDRT(
            in_width=           10,
            in_dropout=         0.1,
            n_layers=           2,
            lay_width=          12,
            dns_scale=          4,
            interlay_dropout=   0.1,
            lay_dropout=        0.1,
            res_dropout=        0.1)
        print(enc_drt)

    def test_EncDRT_init_shared(self):
        enc_drt = EncDRT(10, n_layers=6, shared_lays=True, dns_scale=4)
        print(enc_drt)

    def test_EncDRT(self):

        in_width = 10
        inp = torch.rand(in_width) - 0.5
        print(inp, inp.dtype)

        enc_drt = EncDRT(in_width, n_layers=6, shared_lays=True, dns_scale=4)
        print(enc_drt)
        self.assertTrue(len(list(enc_drt.children())) == 2) # ln_in, shared LayDRT
        out = enc_drt(inp)
        print(out)
        self.assertTrue(out['zeroes'].shape[0] == 6*in_width*4)

        enc_drt = EncDRT(in_width, n_layers=5, lay_width=2*in_width, dns_scale=3)
        print(enc_drt)
        self.assertTrue(len(list(enc_drt.children())) == 7) # projection, ln_in, 5 * LayDRT
        out = enc_drt(inp)
        print(out)
        self.assertTrue(out['zeroes'].shape[0] == 5*2*in_width*3)

        enc_drt = EncDRT(
            in_width=       in_width,
            in_dropout=     0.1,
            n_layers=       4,
            lay_width=      16,
            do_scaled_dns=  True,
            dns_scale=      3,
            lay_dropout=    0.3,
            res_dropout=    0.3)
        print(enc_drt)
        self.assertTrue(len(list(enc_drt.children())) == 7) # in_drop, projection, ln_in, 4 * LayDRT
        out = enc_drt(inp)
        print(out)
        self.assertTrue(out['zeroes'].shape[0] == 4*16*3)

    def test_LayBlockCNN_base_init_call(self):

        n_filters = 6
        lay_cnn = LayBlockCNN(n_filters, do_zeroes=False)
        print(lay_cnn)

        inp = torch.rand(4, n_filters)
        cnn_out = lay_cnn(inp)
        print(cnn_out)

        out = cnn_out['out']
        state = cnn_out['state']
        zeroes = cnn_out['zeroes']
        self.assertTrue(list(out.shape) == [4,6] and state is None and zeroes is None)

        lay_cnn = LayBlockCNN(n_filters)
        cnn_out = lay_cnn(inp)
        zeroes = cnn_out['zeroes']
        self.assertTrue(list(zeroes.shape) == [6])

    def test_LayBlockCNN_no_pad(self):

        n_filters = 6
        lay_cnn = LayBlockCNN(n_filters, padded=False)
        print(lay_cnn)

        inp = torch.rand(4, n_filters)
        cnn_out = lay_cnn(inp)
        out = cnn_out['out']
        print(out.shape)
        self.assertTrue(list(out.shape) == [2,6])

    def test_LayBlockCNN_input_shapes(self):

        n_filters = 6
        lay_cnn = LayBlockCNN(n_filters)
        print(lay_cnn)

        for inp in [
            torch.rand(n_filters),
            torch.rand(4,n_filters),
            torch.rand(5,4,n_filters)
        ]:
            cnn_out = lay_cnn(inp)
            out = cnn_out['out']
            print(out.shape)
            self.assertTrue(out.shape == inp.shape)

    def test_LayBlockCNN_history(self):

        n_filters = 6
        lay_cnn = LayBlockCNN(n_filters)

        history = lay_cnn.get_zero_history()
        print(f'history.shape: {history.shape}')
        self.assertTrue(list(history.shape) == [2,6])

        for inp in [
            torch.rand(n_filters),
            torch.rand(4,n_filters),
            torch.rand(5,4,n_filters),
        ]:
            print('========================')
            print(f'inp.shape: {inp.shape}')
            cnn_out = lay_cnn(inp, history=history)
            out = cnn_out['out']
            state = cnn_out['state']
            print(f'out.shape: {out.shape}')
            print(f'state.shape: {state.shape}')
            cnn_out = lay_cnn(inp, history=state)
            out = cnn_out['out']
            state = cnn_out['state']
            print(f'out.shape: {out.shape}')
            print(f'state.shape: {state.shape}')

    def test_LayBlockCNN_moving_history(self):

        n_filters = 6
        lay_cnn = LayBlockCNN(n_filters)

        history = lay_cnn.get_zero_history()
        print(f'history.shape: {history.shape}')
        print(f'history: {history}')

        state = history
        for ix in range(5):
            print(f'== {ix} ====================')
            inp = torch.rand(1, n_filters)
            print(f'inp: {inp}')
            print(f'state: {state}')
            state_prev = state
            cnn_out = lay_cnn(inp, history=state)
            out = cnn_out['out']
            state = cnn_out['state']
            print(f'out: {out}')
            print(f'state: {state}')
            print(state_prev[1])
            print(state[0])
            self.assertTrue(torch.equal(state_prev[1], state[0]))

    def test_LayBlockCNN_base_encoder(self):

        n_time = 6
        n_filters = 4
        inp = torch.rand(n_time, n_filters)
        print(f'inp shape {inp.shape}')

        for kernel_size in [3,5,7,9]:
            lay_cnn = LayBlockCNN(n_filters, kernel_size=kernel_size)
            print('===============')
            print(lay_cnn)

            out = lay_cnn(inp)
            print(f'out shape {out["out"].shape}')
            print(f'state: {out["state"]}')
            self.assertTrue(inp.shape == out['out'].shape)
            self.assertTrue(out['state'] is None)

    def test_LayBlockCNN_base_casual(self):

        n_time = 6
        n_filters = 4
        inp = torch.rand(n_time, n_filters)
        print(f'inp shape {inp.shape}')

        for kernel_size in [3,5,7,9]:
            lay_cnn = LayBlockCNN(n_filters, kernel_size=kernel_size)
            print(lay_cnn)

            zero_history = lay_cnn.get_zero_history()
            self.assertTrue(list(zero_history.shape) == [kernel_size-1, n_filters])
            print(f'zero_history shape {zero_history.shape}')

            out = lay_cnn(inp, history=zero_history)
            print(f'out shape {out["out"].shape}')
            print(f'state shape: {out["state"].shape}')
            self.assertTrue(out['out'].shape == inp.shape)
            self.assertTrue(out['state'].shape == zero_history.shape)

    def test_LayBlockCNN_kwargs(self):

        n_filters = 4
        inp = torch.rand(6, n_filters)
        print(inp)

        lay_cnn = LayBlockCNN(
            n_filters=      n_filters,
            lay_dropout=    0.1,
            res_dropout=    0.1,
            do_ldrt=        True)
        print(lay_cnn)
        out = lay_cnn(inp)
        print(out)

    def test_LayBlockCNN_more(self):

        n_time = 32
        n_filters = 64
        inp = torch.rand(16, n_time, n_filters)
        print(f'inp shape {inp.shape}')

        kernel_size = 3
        lay_cnn = LayBlockCNN(n_filters, kernel_size=kernel_size)
        print(lay_cnn)

        zero_history = lay_cnn.get_zero_history()
        print(f'zero_history shape {zero_history.shape}')

        out = lay_cnn(inp, history=zero_history)
        print(f'out shape {out["out"].shape}')
        print(f'state shape: {out["state"].shape}')
        self.assertTrue(out['out'].shape == inp.shape)

        o_shape = list(out['out'].shape)
        o_shape[-2] = kernel_size - 1
        self.assertTrue(o_shape == list(out["state"].shape))

    def test_EncCNN_base_init_call(self):

        in_features = 8
        enc = EncCNN(in_features)
        print(enc)

        inp = torch.rand(7, 5, in_features)
        print(f'inp.shape: {inp.shape}')
        enc_out = enc(inp)

        out = enc_out['out']
        state = enc_out['state']
        zeroes = enc_out['zeroes']
        print(f'out.shape: {out.shape}')
        self.assertTrue(list(out.shape) == [7,5,in_features] and state is None and list(zeroes.shape) == [48])

    def test_EncCNN_input_shapes(self):

        in_features = 8
        enc = EncCNN(in_features)

        for inp in [
            torch.rand(in_features),
            torch.rand(5,in_features),
            torch.rand(7,5,in_features)
        ]:
            print(f'inp.shape: {inp.shape}')
            enc_out = enc(inp)
            out = enc_out['out']
            print(f'out.shape: {out.shape}')
            self.assertTrue(out.shape == inp.shape)

    def test_EncCNN_init_raises(self):
        self.assertRaises(TorchnessException, EncCNN, in_features=6, kernel_size=4) # even number for kernel

    def test_EncCNN_no_pad(self):

        in_features = 8
        enc = EncCNN(in_features, padded=False)

        inp = torch.rand(5,20,in_features)
        enc_out = enc(inp)
        out = enc_out['out']
        print(out.shape)
        self.assertTrue(list(out.shape) == [5,8,8])

    def test_EncCNN_history(self):

        in_features = 8
        enc = EncCNN(in_features)

        history = enc.get_zero_history()
        print(f'history.shape: {history.shape}')
        self.assertTrue(list(history.shape) == [6,2,8])

        for inp in [
            torch.rand(in_features),
            torch.rand(5,in_features),
            torch.rand(7,5,in_features)
        ]:
            print('========================')
            print(f'inp.shape: {inp.shape}')
            enc_out = enc(inp, history=history)
            out = enc_out['out']
            state = enc_out['state']
            print(f'out.shape: {out.shape}')
            print(f'state.shape: {state.shape}')
            enc_out = enc(inp, history=state)
            out = enc_out['out']
            state = enc_out['state']
            print(f'out.shape: {out.shape}')
            print(f'state.shape: {state.shape}')

    def test_EncCNN_casual(self):

        in_features = 8
        inp = torch.rand(7,5,in_features)
        print(f'inp.shape: {inp.shape}')

        enc = EncCNN(in_features)

        zero_history = enc.get_zero_history()
        print(f'zero_history.shape: {zero_history.shape}')
        enc_out = enc(inp, history=zero_history)

        out = enc_out['out']
        print(f'out.shape: {out.shape}')
        self.assertTrue(out.shape == inp.shape)

        state = enc_out['state']
        print(f'state.shape: {state.shape}')
        self.assertTrue(list(state.shape) == [7,6,2,8])

    def test_EncCNN_kwargs(self):

        n_filters = 48
        in_features = 32
        inp = torch.rand(18,96,in_features)
        print(inp.shape)

        enc = EncCNN(
            in_features=        in_features,
            time_drop=          0.1,
            feat_drop=          0.2,
            n_layers=           5,
            kernel_size=        7,
            n_filters=          n_filters,
            lay_dropout=        0.15,
            res_dropout=        0.25,
            do_ldrt=            True,
            ldrt_drop=          0.05,
            ldrt_res_dropout=   0.07)
        print(enc)
        enc_out = enc(inp)

        print(enc_out['out'].shape)
        in_sh = list(inp.shape)
        in_sh[-1] = n_filters
        self.assertTrue(list(enc_out['out'].shape) == in_sh)

        zero_history = enc.get_zero_history()
        print(zero_history.shape)
        self.assertTrue(list(zero_history.shape) == [5,6,48])

    def test_EncCNN_shared(self):

        self.assertRaises(TorchnessException, EncCNN, in_features=6, kernel_size=4) # even number for kernel

        in_features = 128
        inp = torch.rand(256,512,in_features)
        print(inp.shape)

        enc = EncCNN(in_features, shared_lays=True)
        print(enc)
        enc_out = enc(inp)

        print(enc_out['out'].shape)
        self.assertTrue(list(enc_out['out'].shape) == list(inp.shape))

        zero_history = enc.get_zero_history()
        print(zero_history.shape)
        self.assertTrue(list(zero_history.shape) == [6,2,128])

    def test_LayBlockTNS_base(self):

        in_features = 64
        inp = torch.rand(16,32,in_features)
        print(inp.shape)

        lay_tns = LayBlockTNS(d_model=in_features)
        print(lay_tns)
        out = lay_tns(inp)
        print(out['out'].shape)
        print(out['zeroes'].shape)

        # TAT
        query = torch.mean(inp, dim=-2, keepdim=True)
        print(query.shape)
        out = lay_tns(inp, task_query=query)
        print(out['out'].shape)
        print(out['zeroes'].shape)

    def test_EncTNS_base(self):

        in_features = 64
        enc = EncTNS(num_layers=4, d_model=in_features)
        print(enc)
        self.assertTrue(enc.pos_emb is None)

        inp = torch.rand(16,32,in_features)
        print(f'inp.shape: {inp.shape}')
        enc_out = enc(inp)
        print(list(enc_out.keys()))
        out = enc_out['out']
        zeroes = enc_out['zeroes']
        print(f'out.shape: {out.shape}')
        print(f'zeroes.shape: {zeroes.shape}')
        self.assertTrue(list(out.shape) ==  [16,32,in_features] and list(zeroes.shape) == [1024])

    def test_EncTNS_input_shapes(self):

        in_features = 16
        enc = EncTNS(num_layers=4, d_model=in_features)

        for inp in [
            torch.rand(in_features),
            torch.rand(5,in_features),
            torch.rand(7,5,in_features),
        ]:
            print('========================')
            print(f'inp.shape: {inp.shape}')
            enc_out = enc(inp)
            out = enc_out['out']
            zeroes = enc_out['zeroes']
            print(f'out.shape: {out.shape}')
            print(f'zeroes.shape: {zeroes.shape}')

    def test_EncTNS_PE(self):

        in_features = 64
        seq_len = 32
        inp = torch.rand(16,seq_len,in_features)
        print(inp.shape)

        enc = EncTNS(num_layers=2, d_model=in_features, max_seq_len=48)
        print(enc)
        print(enc.pos_emb)
        self.assertTrue(enc.pos_emb is not None)
        out = enc(inp)
        print(out['out'].shape)

    def test_EncTNS_TAT(self):

        in_features = 64
        seq_len = 32
        inp = torch.rand(16, seq_len, in_features)
        print(inp.shape)

        enc = EncTNS(
            num_layers=     4,
            num_layers_TAT= 2,
            d_model=        in_features)
        print(enc)
        out = enc(inp)
        print(out['out'].shape)

    def test_EncTNS_TAT_shared(self):

        in_features = 64
        seq_len = 32
        inp = torch.rand(16, seq_len, in_features)
        print(inp.shape)

        enc = EncTNS(
            num_layers=     4,
            num_layers_TAT= 2,
            shared_lays=    (3,1,2),
            d_model=        in_features)
        print(enc)
        out = enc(inp)
        print(out['out'].shape)
