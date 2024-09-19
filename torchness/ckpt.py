from collections import OrderedDict
import torch
from typing import Optional

from torchness.initialize import my_initializer
from torchness.base import NUM


def ckpt_nfo(
        ckptA: str,                     # checkpoint A (file name)
        ckptB: Optional[str]=   None,   # checkpoint B (file name)
):
    """ returns checkpoint info, if given two - checks if B is equal A """

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


def mrg_ckpts(
        ckptA: str,             # checkpoint A (file name)
        ckptB: Optional[str],   # checkpoint B (file name), for None takes 100% ckptA
        ckptM: str,             # checkpoint merged (file name)
        ratio: NUM=     0.5,    # ratio of merge
        noise: NUM=     0.0,    # noise factor, amount of noise added to new value <0.0;1.0>
):
    """ weighted merge of two checkpoints (on CPU)
    does NOT check for compatibility of two checkpoints, but will crash if those are not compatible
    forced to perform on CPU device (not to raise any CUDA errors) """
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
