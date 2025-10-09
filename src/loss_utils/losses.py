import torch
from math import exp, log, pi

def l1(x, y):
    return torch.abs(x-y)


def l2(x, y):
    return ((x-y)**2)


def penalty_loss(occupancy_values, epsilon):
    # Compute penalty: max(0, epsilon - occupancy_values)
    penalty = torch.clamp(epsilon - occupancy_values, min=0)
    
    # Average the penalties across all points
    total_penalty = penalty.mean()
    return total_penalty

def or_loss_directed(network_output, gt, confs = None, weight = None, mask = None, type='min'):

    weight = torch.ones_like(gt[:1]) if weight is None else weight
    
    if type == 'min':
        loss = torch.minimum(
            (network_output - gt).abs(),
            torch.minimum(
                (network_output - gt - 1).abs(), 
                (network_output - gt + 1).abs()
            ))
    else:
        loss = (network_output - gt).abs()

    loss = loss * pi
    if confs is not None:
        loss = loss * confs - (confs + 1e-7).log()    
    if mask is not None:
        loss = loss * mask
    if weight is not None:
        return (loss * weight).sum() / weight.sum()
    else:
        return loss * weight
    


def l2_loss(network_output, gt, weight = None, mask = None):
    loss = (network_output - gt).pow(2)
    if mask is not None:
        loss = loss * mask
    if weight is not None:
        return (loss * weight).sum() / weight.sum()
    else:
        return loss.mean()
    