import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.append(os.getcwd())

from __Utils import config

def dice_loss(inputs, targets, eps=1e-7):
    inputs = torch.sigmoid(inputs)
    
    #flatten label and prediction tensors
    targets = torch.eye(inputs.shape[1])[targets].permute(0,3,1,2).float().to(config.DEVICE)
    
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection)/(inputs.sum() + targets.sum() + eps)  
    
    return 1 - dice


def dice_loss_per_batch(output, target, eps=1e-7):
    eps = 1e-7
    # convert target to onehot
    targ_onehot = torch.eye(output.shape[1])[target].permute(0,3,1,2).float().to(config.DEVICE)
    # convert logits to probs
    pred = F.softmax(output, dim=1)
    # sum over HW
    inter = (pred * targ_onehot).sum(axis=[0,2,3])
    union = (pred + targ_onehot).sum(axis=[0,2,3])
    # mean over C
    dice_sum = (2. * inter / (union + eps)).sum()
    non_zero = (union != 0).sum()
    dice = dice_sum / non_zero # mean for classes that are in target or are wrongly predicted
    return 1. - dice


def dice_loss_per_image(output, target, eps=1e-7):
    eps = 1e-7
    # convert target to onehot
    targ_onehot = torch.eye(output.shape[1])[target].permute(0,3,1,2).float().to(config.DEVICE)
    # convert logits to probs
    pred = F.softmax(output, dim=1)
    # sum over HW
    inter = (pred * targ_onehot).sum(axis=[2,3])
    union = (pred + targ_onehot).sum(axis=[2,3])
    # mean over C
    dice = (2. * inter / (union + eps)).sum(axis=1)
    non_zero = (union != 0).sum(axis=1)
    dice = dice / non_zero # mean for classes that are in target or are wrongly predicted
    return 1. - dice.mean()
    

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, output, targ):
        return dice_loss_per_batch(output, targ)

    def activation(self, output):
        return F.softmax(output, dim=1)
    
    def decodes(self, output):
        return output.argmax(1)