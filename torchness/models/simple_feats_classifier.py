import torch
from typing import Tuple, Optional

from torchness.motorch import Module
from torchness.base import INI, TNS, DTNS
from torchness.initialize import my_initializer
from torchness.layers import LayDense
from torchness.encoders import EncDRT


class SFeatsCSF(Module):
    """ Simple Feats Classification Module """

    def __init__(
            self,
            feats_width: int,                               # input
            in_lay_norm: bool=                      True,
            in_dropout: float=                      0.0,
            n_hidden: int=                          2,
            hidden_width: int=                      12,
            hidden_drop: float=                     0.0,
            num_classes: int=                       2,      # output
            lay_norm=                               True,
            do_zeroes: bool=                        True,
            class_weights: Optional[Tuple[float]]=  None,
            initializer: INI=                       None,
            **kwargs):

        super().__init__(**kwargs)

        self.logger.info(f'*** SFeatsCSF (Module) *** inits for feats of width {feats_width}')

        if initializer is None:
            initializer = my_initializer

        self.enc_drt = EncDRT(
            in_width=           feats_width,
            in_lay_norm=        in_lay_norm,
            in_dropout=         in_dropout,
            n_layers=           n_hidden,
            lay_width=          hidden_width,
            do_scaled_dns=      False,
            activation=         torch.nn.ReLU,
            interlay_dropout=   0.0,
            lay_dropout=        hidden_drop,
            res_dropout=        0.0,
            lay_norm=           lay_norm,
            do_zeroes=          do_zeroes,
            initializer=        initializer)

        self.logits = LayDense(
            in_features=    hidden_width,
            out_features=   num_classes,
            activation=     None,
            bias=           False,
            initializer=    initializer)

        if class_weights:
            class_weights = torch.nn.Parameter(torch.tensor(class_weights), requires_grad=False)
        self.class_weights = class_weights

    def forward(self, feats:TNS) -> DTNS:
        enc_out = self.enc_drt(feats)
        logits = self.logits(enc_out['out'])
        dist = torch.distributions.Categorical(logits=logits)
        return {
            'logits':   logits,
            'entropy':  dist.entropy(),
            'probs':    dist.probs,
            'preds':    torch.argmax(logits, dim=-1),
            'zeroes':   enc_out['zeroes']}

    def loss(self, feats:TNS, labels:TNS) -> DTNS:

        out = self.forward(feats)
        logits = out['logits']

        loss = torch.nn.functional.cross_entropy(
            input=      logits,
            target=     labels,
            weight=     self.class_weights,
            reduction=  'mean')
        acc = self.accuracy(logits=logits, labels=labels)
        f1 = self.f1(logits=logits, labels=labels)
        out.update({
            'loss': loss,
            'acc':  acc,
            'f1':   f1})
        return out