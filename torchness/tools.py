import torch
from typing import Tuple

from torchness.base import TNS, DTNS


def min_max_probs(probs:TNS) -> DTNS:
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


def normalize_logits(logits:TNS) -> TNS:
    """ normalizes N-dim log-prob tensor """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.log(probs)
    # or this way:
    # return logits - logits.logsumexp(dim=-1, keepdim=True)


### scores *****************************************************************************************

def cross_entropy_loss(logits:TNS, target:TNS) -> TNS:
    """ cross-entropy loss for:
    N-dim log-prob
    N-1-dim target of indexes (int) """
    return torch.nn.functional.cross_entropy(input=logits, target=target)


def perplexity(logits:TNS, target:TNS) -> TNS:
    """ perplexity of:
    N-dim log-prob
    N-1-dim target of indexes (int) """
    logits_norm = normalize_logits(logits)
    action_target_logits_mean = logits_norm[range(len(logits)), target].mean()
    ppx = torch.exp(-action_target_logits_mean)
    # or this way:
    # ce_loss = cross_entropy_loss(logits, action_target)
    # ppx = torch.exp(ce_loss)
    return ppx


def accuracy(pred:TNS, target:TNS) -> TNS:
    return (pred == target).to(torch.float).mean()


def brier_score(probs:TNS, action_sampled:TNS, action_target:TNS):
    """ Brier score for distribution modeling 
    probs - N-dim 
    action sampled - N-1-dim of indexes (int) 
    action target - N-1-dim of indexes (int) """
    y = (action_sampled == action_target).to(torch.float)
    return torch.pow(probs[range(len(action_target)), action_target] - y, 2).mean()


def mean_square_error(pred:TNS, target:TNS):
    """ MSE of N-dim pred and N-dim target """
    return torch.nn.functional.mse_loss(input=pred, target=target)


def diff_avg_max(pred:TNS, target:TNS) -> Tuple[TNS,TNS,TNS]:
    """ differences (similar to MSE) of N-dim pred and N-dim target
    returns:
    probs diff avg
    probs diff max avg
    probs diff max """
    diff = torch.abs(pred - target)
    return diff.mean(), diff.max(dim=-1)[0].mean(), diff.max()
