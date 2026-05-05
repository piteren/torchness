import torch

from torchness.layers import LayDense, TF_Dropout, LayConv1D, PositionalEncoding, zeroes


def test_lay_dense():
    tns = torch.rand(20)
    print(tns)

    dnsl = LayDense(20, 10)
    print(dnsl)
    out = dnsl(tns)
    print(out)
    assert out.size()[0] == 10
    print(torch.sum(out))
    assert float(torch.sum(out)) >= 0.0

    dnsl = LayDense(
        in_features=    20,
        out_features=   10,
        activation=     None,
        bias=           False,
        initializer=    torch.nn.init.zeros_)
    out = dnsl(tns)
    print(out)
    print(torch.sum(out))
    assert float(torch.sum(out)) == 0.0


def test_tf_dropout():
    tns = torch.rand((5, 5, 5))
    sum_tns = float(torch.sum(tns))
    print(sum_tns)
    drl = TF_Dropout(time_drop=0.3, feat_drop=0.3)
    dropped = drl(tns)
    print(dropped)
    sum_dropped = float(torch.sum(dropped))
    print(sum_dropped)


def test_tf_dropout_higher_dim():
    tns = torch.rand((6, 5, 12, 14))
    sum_tns = float(torch.sum(tns))
    print(sum_tns)
    drl = TF_Dropout(time_drop=0.3, feat_drop=0.6)
    dropped = drl(tns)
    print(dropped.shape)
    sum_dropped = float(torch.sum(dropped))
    print(sum_dropped)


def test_LayConv1D():
    inp = torch.rand(3, 6)  # [Channels,SignalSeq]
    print(inp.shape, inp)
    conv_lay = LayConv1D(6, 8)
    print(conv_lay)
    out = conv_lay(inp)
    print(out.shape, out)
    assert tuple(out.shape) == (3, 8)


def test_PositionalEncoding():
    tns = torch.zeros(12, 32, 64)
    print(tns)
    print(tns.shape)
    pe = PositionalEncoding(64)
    out = pe(tns)
    print(out)
    print(out.shape)


def test_zeroes():
    tns = torch.rand(5)
    z = zeroes(tns)
    print(z, z.shape)
    assert z.shape[-1] == 5

    tns = torch.rand((3, 4, 5))
    z = zeroes(tns)
    print(z, z.shape)
    assert z.shape[-1] == 5

    drl = TF_Dropout(time_drop=0.3, feat_drop=0.3)
    dropped = drl(tns)
    print(dropped)
    print(zeroes(dropped))
