from collections import OrderedDict
import torch
from typing import Optional

from torchness.types import NUM, TNS, DTNS



class TorchnessException(Exception):
    pass

# weights initializer from BERT, the only difference is that in torch values are CLAMPED not SAMPLED till in <a,b>
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


def mrg_ckpts(
        ckptA: str,             # checkpoint A (file name)
        ckptB: Optional[str],   # checkpoint B (file name), for None takes 100% ckptA
        ckptM: str,             # checkpoint merged (file name)
        ratio: NUM=     0.5,    # ratio of merge
        noise: NUM=     0.0,    # noise factor, amount of noise added to new value <0.0;1.0>
):
    """
    weighted merge of two checkpoints (on CPU)
    does NOT check for compatibility of two checkpoints, but will crash if those are not compatible
    forced to perform on CPU device (not to raise any CUDA errors)
    """
    with torch.no_grad():

        checkpointA = torch.load(ckptA, map_location='cpu')
        checkpointB = torch.load(ckptB, map_location='cpu') if ckptB else checkpointA

        cmsdA = checkpointA['model_state_dict']
        cmsdB = checkpointB['model_state_dict']
        cmsdM = OrderedDict()

        for k in cmsdA:
            tnsA = cmsdA[k]
            if tnsA.is_floating_point():

                if type(ratio) is torch.Tensor:
                    ratio = ratio.cpu()

                tnsB = cmsdB[k]
                cmsdM[k] = ratio * tnsA + (1 - ratio) * tnsB

                if noise > 0.0:

                    std_dev = torch.std(tnsA)
                    if std_dev != 0.0:

                        noise_tensor = torch.zeros_like(tnsA)
                        my_initializer(noise_tensor, std=std_dev)

                        if type(noise) is torch.Tensor:
                            noise = ratio.cpu()

                        cmsdM[k] += noise * noise_tensor
            else:
                cmsdM[k] = tnsA

        checkpoint_M = {}
        checkpoint_M.update(checkpointA)
        checkpoint_M['model_state_dict'] = cmsdM

        torch.save(checkpoint_M, ckptM)

# returns base checkpoint information, if given two - checks if B is equal A
def ckpt_nfo(
        ckptA: str,                     # checkpoint A (file name)
        ckptB: Optional[str]=   None,   # checkpoint B (file name)
):
    checkpoint_A = torch.load(ckptA, map_location='cpu')
    checkpoint_B = torch.load(ckptB, map_location='cpu') if ckptB else None
    are_equal = True

    cmsd_A = checkpoint_A['model_state_dict']
    cmsd_B = checkpoint_B['model_state_dict'] if checkpoint_B else None

    print(f'Checkpoint has {len(cmsd_A)} tensors, #floats: {sum([cmsd_A[k].numel() for k in cmsd_A])}')
    for k in cmsd_A:
        tns = cmsd_A[k]
        print(f'{k:100} shape: {str(list(tns.shape)):15} {tns.dtype}')
        if cmsd_B:
            if k in cmsd_B:
                if not torch.equal(cmsd_A[k], cmsd_B[k]):
                    print(f' ---> is not equal in second checkpoint')
                    are_equal = False
            else:
                print(f' ---> is not present in second checkpoint')
                are_equal = False
    if checkpoint_B:
        print(f'Checkpoints {"are equal" if are_equal else "are NOT equal"}')


def min_max_probs(probs: TNS) -> DTNS:
    with torch.no_grad():
        max_probs = torch.max(probs, dim=-1)[0] # max probs
        min_probs = torch.min(probs, dim=-1)[0] # min probs
        max_probs_mean = torch.mean(max_probs)  # mean of max probs
        min_probs_mean = torch.mean(min_probs)  # mean of min probs
    return {'max_probs_mean':max_probs_mean, 'min_probs_mean':min_probs_mean}


def select_with_indices(source:TNS, indices:TNS) -> TNS:
    """ selects from the (multidimensional dim) source
    values from the last axis
    given with indices (dim-1) tensor of ints """
    indices = torch.unsqueeze(indices, dim=-1)
    source_selected = torch.gather(source, dim=-1, index=indices)
    return torch.squeeze(source_selected, dim=-1)
