import logging
import math
import numpy as np
from tqdm import tqdm

from torchness.base import INI, TNS, DTNS
from torchness.motorch import Module, MOTorch
from torchness.models.text_embbeder import TextEMB
from torchness.models.simple_feats_classifier import SFeatsCSF

logger = logging.getLogger(__name__)


class STextCSF(Module):
    """ Simple Text Classification Module, based on Sentence-Transformer """

    def __init__(
            self,
            st_name: str=                           'all-MiniLM-L6-v2',
            enc_batch_size=                         256,
            in_dropout: float=                      0.0,
            hidden_width: int=                      30,
            hidden_dropout: float=                  0.0,
            num_classes: int=                       2,
            class_weights: list[float] | None = None,
            initializer: INI=                       None,
            **kwargs):

        super().__init__(**kwargs)

        self.te_module = TextEMB(
            st_name=        st_name,
            enc_batch_size= enc_batch_size)

        self.csf = SFeatsCSF(
            feats_width=    self.te_module.width,
            in_dropout=     in_dropout,
            hidden_width=   hidden_width,
            hidden_dropout= hidden_dropout,
            num_classes=    num_classes,
            class_weights=  class_weights,
            initializer=    initializer)

    def encode(
            self,
            texts: str | list[str],
            show_progress_bar=  'auto',
            device=             None,
    ) -> np.ndarray:
        if show_progress_bar == 'auto':
            show_progress_bar = False
            if type(texts) is list and len(texts) > 1000:
                show_progress_bar = True
        return self.te_module.encode(
            texts=              texts,
            show_progress_bar=  show_progress_bar,
            device=             device)

    def forward(self, feats:TNS) -> DTNS:
        return self.csf(feats)

    def loss(self, feats:TNS, labels:TNS) -> DTNS:
        return self.csf.loss(feats, labels)


class STextCSF_MOTorch(MOTorch):

    def __init__(
            self,
            module_type: type[STextCSF] | None = STextCSF,
            enc_batch_size=                         128,    # number of lines in batch for embeddings
            fwd_batch_size=                         256,    # number of embeddings in batch for probs
            **kwargs):

        super().__init__(
            module_type=    module_type,
            enc_batch_size= enc_batch_size,
            fwd_batch_size= fwd_batch_size,
            **kwargs)

    def get_embeddings(
            self,
            lines: str | list[str],
            show_progress_bar=      'auto',
    ) -> np.ndarray:
        if type(lines) is str: lines = [lines]
        logger.info(f'{self.name} prepares embeddings for {len(lines)} lines ..')
        if show_progress_bar == 'auto':
            show_progress_bar = logger.level < 21 and len(lines) > 1000
        return self.module.encode(
            texts=              lines,
            show_progress_bar=  show_progress_bar,
            device=             self.device) # needs to give device here because of SentenceTransformer bug in encode() #153

    def get_probs(self, lines:str | list[str]) -> np.ndarray:

        embs = self.get_embeddings(lines)

        num_splits = math.ceil(embs.shape[0] / self['fwd_batch_size']) # INFO: gives +- batch_size
        featsL = np.array_split(embs,num_splits)

        logger.info(f'{self.name} computes probs for {len(featsL)} batches of embeddings')
        iter = tqdm(featsL) if logger.level < 21 else featsL
        probs = np.concatenate([self(feats)['probs'].detach().cpu().numpy() for feats in iter])
        logger.info(f'> got probs {probs.shape}')

        return probs

    def get_probsL(self, linesL: list[list[str]]) -> list[np.ndarray]:

        lines = []
        for l in linesL:
            lines += l

        probs = self.get_probs(lines)

        acc_lengths = []
        acc = 0
        for l in [len(ls) for ls in linesL]:
            acc_lengths.append(l+acc)
            acc += l
        acc_lengths.pop(-1)

        if acc_lengths: return np.split(probs,acc_lengths)
        else:           return [probs]
