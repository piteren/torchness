import math
import torch
from typing import Optional

from torchness.base import ACT, INI, TNS, TorchnessException
from torchness.initialize import my_initializer


class LayDense(torch.nn.Linear):
    """ my dense layer, linear with initializer + activation """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: ACT=    torch.nn.ReLU,
            bias: bool=         True,
            initializer: INI=   None,
            **kwargs):
        self.initializer = initializer or my_initializer
        super().__init__(
            in_features=    in_features,
            out_features=   out_features,
            bias=           bias,
            **kwargs)
        self.activation = activation() if activation else None

    def reset_parameters(self) -> None:

        self.initializer(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

            ### original Linear (with uniform) reset for bias
            # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inp:TNS) -> TNS:
        out = super().forward(inp)
        if self.activation: out = self.activation(out)
        return out

    def extra_repr(self) -> str:
        act_info = '' if self.activation else ', activation=None'
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}{act_info}'


class TF_Dropout(torch.nn.Dropout):
    """ Time & Feats Dropout -> for sequences
    general pattern od inp tensor shape: [batch, seq, feats] """

    def __init__(
            self,
            time_drop: float=   0.0,
            feat_drop: float=   0.0,
            inplace: bool=      False):
        self.time_drop = time_drop
        self.feat_drop = feat_drop
        super(TF_Dropout, self).__init__(inplace=inplace)

    def forward(self, inp:TNS) -> TNS:

        output = inp
        in_shape = inp.size()

        if self.time_drop:
            t_drop = torch.ones(in_shape[-2])
            t_drop = torch.nn.functional.dropout(
                input=      t_drop,
                p=          self.time_drop,
                training=   self.training,
                inplace=    self.inplace)
            t_drop = torch.unsqueeze(t_drop, dim=-1)
            output = output * t_drop

        if self.feat_drop:
            f_drop = torch.ones(in_shape[-1])
            f_drop = torch.nn.functional.dropout(
                input=      f_drop,
                p=          self.feat_drop,
                training=   self.training,
                inplace=    self.inplace)
            f_drop = torch.unsqueeze(f_drop, dim=-2)
            output = output * f_drop

        return output

    def extra_repr(self) -> str:
        return f'time_drop={self.time_drop}, feat_drop={self.feat_drop}, inplace={self.inplace}'


class LayConv1D(torch.nn.Conv1d):
    """ my Conv1D, with initializer + activation """

    def __init__(
            self,
            in_features: int,                   # input num of channels
            n_filters: int,                     # output num of channels
            kernel_size: int=   3,
            stride=             1,              # single number or a one-element tuple
            padding=            'same',
            dilation=           1,
            groups=             1,
            bias=               True,
            padding_mode=       'zeros',
            activation: ACT=    torch.nn.ReLU,
            initializer: INI=   None,
            **kwargs):

        super(LayConv1D, self).__init__(
            in_channels=    in_features,
            out_channels=   n_filters,
            kernel_size=    kernel_size,
            stride=         stride,
            padding=        padding,
            dilation=       dilation,
            groups=         groups,
            bias=           bias,
            padding_mode=   padding_mode,
            **kwargs)

        self.activation = activation() if activation else None

        if not initializer: initializer = my_initializer
        initializer(self.weight)
        if self.bias is not None:
            # original Conv1D (with uniform) reset for bias
            # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # torch.nn.init.uniform_(self.bias, -bound, bound)
            torch.nn.init.zeros_(self.bias)

    def forward(self, inp:TNS) -> TNS:
        inp_trans = torch.transpose(input=inp, dim0=-1, dim1=-2) # transposes inp to (N,C,L) <- (N,L,C), since torch.nn.Conv1d assumes that channels is @ -2 dim
        out = super().forward(input=inp_trans)
        out = torch.transpose(out, dim0=-1, dim1=-2) # transpose back
        if self.activation: out = self.activation(out)
        return out


class LayRES(torch.nn.Module):
    """ Residual Layer with dropout for bypass inp """

    def __init__(
            self,
            in_features: Optional[int]= None,
            dropout: float=             0.0):

        if dropout and in_features is None:
            raise TorchnessException('LayRES with dropout needs to know its in_features (int) - cannot be None')

        super(LayRES, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout) if dropout else None

    def forward(self, inp:TNS, bypass:TNS) -> TNS:
        if self.dropout:
            bypass = self.dropout(bypass)
        return inp + bypass


class PositionalEncoding(torch.nn.Module):

    def __init__(
            self,
            d_model: int,
            dropout: float= 0.0,
            max_len: int=   512):

        super().__init__()

        self.dropout = torch.nn.Dropout(p=dropout) if dropout else None

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    # x - tensor of shape [..,seq,feats]
    def forward(self, x:TNS) -> TNS:
        x = x + self.pe[:x.size(-2)]
        if self.dropout:
            x = self.dropout(x)
        return x


def zeroes(inp:TNS, no_grad=True) -> TNS:
    """ returns [0,1] Tensor: 1 where inp not activated (value =< 0)
    looks at last dimension / features """

    was_grad_enabled = False
    if no_grad and torch.is_grad_enabled():
        torch.set_grad_enabled(False)
        was_grad_enabled = True

    activated = (inp > 0).to(int)
    axes = list(range(len(inp.shape)))[:-1]  # all but last(feats) axes indexes list like: [0,1,2] for 4d shape
    activated_reduced = torch.sum(activated, dim=axes) if axes else activated  # 1 or more for activated, 0 for not activated, if not axes -> we have only-feats-tensor-case
    zs = (activated_reduced == 0).to(int)

    if was_grad_enabled:
        torch.set_grad_enabled(True)

    return zs