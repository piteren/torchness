import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

from torchness.motorch import MOTorch, Module


class TextEMB(Module):
    """ Text Embedding Module
    based on Sentence-Transformer
    prepares embedding for given text (str) """

    def __init__(
            self,
            st_name: str=   'all-MiniLM-L6-v2',
            enc_batch_size= 256,
            **kwargs):
        super().__init__(**kwargs)
        self.st_name = st_name
        self.st_model = SentenceTransformer(model_name_or_path=st_name)
        self.enc_batch_size = enc_batch_size
        self.logger.info(f'*** TextEMB : {self.st_name} *** initialized, feats width:{self.width} seq length:{self.length}')

    def tokenize(self, texts:Union[str,List[str]]) -> Union[List[str], List[List[str]]]:
        tokenizer = self.st_model.tokenizer
        if type(texts) is str:
            return tokenizer.tokenize(texts)
        return [tokenizer.tokenize(t) for t in texts]

    # original, wrapped version
    def encode(
            self,
            texts: Union[str,List[str]],
            show_progress_bar=  True,
            device=             None,
    ) -> np.ndarray:
        return self.st_model.encode(
            sentences=          texts,
            batch_size=         self.enc_batch_size,
            show_progress_bar=  show_progress_bar,
            device=             device)

    @property
    def width(self) -> int:
        return self.st_model.get_sentence_embedding_dimension()

    @property
    def length(self) -> int:
        return self.st_model.get_max_seq_length()


# is MOTorch for given st_name (SentenceTransformer) based on TextEMB module
class TextEMB_MOTorch(MOTorch):

    def __init__(self, module_type:type(TextEMB)=TextEMB, **kwargs):
        super().__init__(module_type=module_type, **kwargs)

    def get_tokens(self, lines:Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        self.logger.info(f'{self.name} prepares tokens for {len(lines)} lines..')
        return self.module.tokenize(lines)

    def get_embeddings(
            self,
            lines: Union[str, List[str]],
            show_progress_bar=  'auto') -> np.ndarray:

        if show_progress_bar == 'auto':
            show_progress_bar = False
            if self.logger.level < 21 and type(lines) is list and len(lines) > 1000:
                show_progress_bar = True

        self.logger.info(f'{self.name} prepares embeddings for {len(lines)} lines..')
        return self.module.encode(
            texts=              lines,
            device=             self.device, # fixes bug of SentenceTransformers.encode() device placement
            show_progress_bar=  show_progress_bar)

    @property
    def width(self) -> int:
        return self.module.width