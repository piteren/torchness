import torch

from torchness.base import NUM


# weights initializer from BERT
# the only difference is that in torch values are CLAMPED not SAMPLED till in <a,b>
def bert_initializer(*args, std:NUM=0.02, **kwargs):
    return torch.nn.init.trunc_normal_(*args, **kwargs, std=std, a=-2*std, b=2*std)


def my_initializer(*args, std:NUM=0.02, **kwargs):
    # different layers use different initialization functions:
    # torch Linear % Conv1D uses kaiming_uniform_(weights) & xavier_uniform_(bias)
    # my TF uses trunc_normal_(weights, std=0.02, a==b==2*std) & 0(bias) <- from BERT
    # - kaiming_uniform_ is uniform_ with bound from 2015 paper, (for relu)
    # - xavier_uniform_ is uniform_ whit bound from 2010 paper (for linear / sigmoid)
    # - trunc_normal_ is normal with mean 0 and given std, all values SAMPLED till in <a,b>
    return bert_initializer(*args, **kwargs, std=std)
