import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from utils import save_checkpoint, load_checkpoint, save_some_examples, eval_some_examples
import config

from dataset import YooxDatasetSegmentation
from model_generator import UNet

import os, sys
sys.path.append(os.getcwd())
from __Utils.DiceLoss import DiceLoss
from __Utils.CrossEntropyLoss import CrossEntropyLoss

from tqdm import tqdm


def main():
    model = UNet(3,6).to(config.DEVICE)

    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    
    load_checkpoint(config.CHECKPOINT_GEN, model, opt, config.LEARNING_RATE)
    

    import cv2
    img = cv2.imread("_0_Segmentation/galan_white.png").transpose(2,0,1)
    model.eval()
    y = model(torch.tensor(img)[None].float().cuda()/255)
    y = torch.argmax(y, 1)

    y = y[0].long().cpu().numpy()
    old_idx = [0, 128, 16384, 32768, 32896, 4194304, 4194432, 4227072, 4227200, 8388608, 8404992, 8421376, 12582912, 12583040, 12615680, 12615808]
    for i in range(len(old_idx)):
        y[y == i] = old_idx[i]
    b = y % 256
    g = y // 256 % 256
    r = y // 256 // 256
    y = np.stack([b,g,r], 0)
    y = y.transpose(1,2,0)

    cv2.imwrite("_0_Segmentation/segmented.png", y)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()