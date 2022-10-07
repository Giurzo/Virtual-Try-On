import torch
import torch.nn as nn
import torch.nn.functional as F

from __Utils import config

def cross_entropy_loss_per_batch(output, target, eps=1e-7):
    eps = 1e-7
    # convert target to onehot
    targ_onehot = torch.eye(output.shape[1])[target].permute(0,3,1,2).float().to(config.DEVICE)
    # convert logits to probs
    pred = F.softmax(output, dim=1)

    pixel_wise_CE = - targ_onehot * torch.log(pred + eps)
    label_sum = pixel_wise_CE.sum(axis=[0,2,3])
    label_count = targ_onehot.sum(axis=[0,2,3])

    label_mean = label_sum / (label_count + eps)

    label_mean = label_mean.sum()
    non_zero = (label_count != 0).sum()
    label_mean = label_mean / non_zero
    return label_mean


def cross_entropy_loss_per_image(output, target, eps=1e-7):
    eps = 1e-7
    # convert target to onehot
    targ_onehot = torch.eye(output.shape[1])[target].permute(0,3,1,2).float().to(config.DEVICE)
    # convert logits to probs
    pred = F.softmax(output, dim=1)

    pixel_wise_CE = - targ_onehot * torch.log(pred + eps)
    label_sum = pixel_wise_CE.sum(axis=[2,3])
    label_count = targ_onehot.sum(axis=[2,3])
    label_mean = label_sum / (label_count + eps)

    label_mean = label_mean.sum(axis=1)
    non_zero = (label_count != 0).sum(axis=1)
    label_mean = label_mean / non_zero
    return label_mean.mean()
    

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, output, targ):
        return cross_entropy_loss_per_batch(output, targ)

    def activation(self, output):
        return F.softmax(output, dim=1)
    
    def decodes(self, output):
        return output.argmax(1)