import os
import random
import numpy as np
import torch
import config
from tqdm import tqdm
import cv2
import time

import os, sys
sys.path.append(os.getcwd())
from __Utils.utils import *
from __Utils.DiceLoss import DiceLoss
from __Utils.CrossEntropyLoss import CrossEntropyLoss


def save_some_examples(model, val_loader):
    create_dir("results/seg")

    size = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)

    model = model.to(config.DEVICE)
    model.eval()
    
    time_taken = []

    for idx, (x, y) in enumerate(val_loader):
        x = x.to(config.DEVICE).float()

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.argmax(pred_y, 1)

            total_time = time.time() - start_time
            time_taken.append(total_time)

        pred_y = pred_y[0].long().cpu().numpy()

        old_idx = [0, 128, 16384, 32768, 32896, 4194304, 4194432, 4227072, 4227200, 8388608, 8404992, 8421376, 12582912, 12583040, 12615680, 12615808]
        for i in range(len(old_idx)):
            pred_y[pred_y == i] = old_idx[i]
        b = pred_y % 256
        g = pred_y // 256 % 256
        r = pred_y // 256 // 256
        pred_y = np.stack([b,g,r], 0)
        pred_y = pred_y.transpose(1,2,0)


        y = y[0].long().cpu().numpy()

        old_idx = [0, 128, 16384, 32768, 32896, 4194304, 4194432, 4227072, 4227200, 8388608, 8404992, 8421376, 12582912, 12583040, 12615680, 12615808]
        for i in range(len(old_idx)):
            y[y == i] = old_idx[i]
        b = y % 256
        g = y // 256 % 256
        r = y // 256 // 256
        y = np.stack([b,g,r], 0)
        y = y.transpose(1,2,0)


        x = x.cpu()[0].numpy().transpose(1,2,0)*255
    
        res = np.concatenate([x, pred_y, y], axis=1)

        cv2.imwrite(f"results/seg/{idx}.png", res.astype(np.uint8))

        if idx > 50:
            break


def eval_some_examples(model, val_loader):

    CELoss = CrossEntropyLoss()
    DICELoss = DiceLoss()

    model = model.to(config.DEVICE)
    model.eval()
    
    time_taken = []

    ce = []
    dice = []

    for idx, (x, y) in enumerate(val_loader):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)

            ce += [ CELoss(pred_y, y).item() ]
            dice += [ DICELoss(pred_y, y).item() ]

            total_time = time.time() - start_time
            time_taken.append(total_time)


        if idx > 50:
            break
    
    print(f"CE : {mean(ce)}")
    print(f"DICE : {mean(dice)}")

    return mean(ce) + mean(dice)