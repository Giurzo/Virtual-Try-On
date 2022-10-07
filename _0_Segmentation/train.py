import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from utils import save_checkpoint, load_checkpoint, save_some_examples, eval_some_examples, mean
import config
from dataset import YooxDatasetSegmentation
from model_generator import UNet

import os, sys
sys.path.append(os.getcwd())
from __Utils.DiceLoss import DiceLoss
from __Utils.CrossEntropyLoss import CrossEntropyLoss

from tqdm import tqdm


def train(model, loader, opt, CELoss, DICELoss):
    loop = tqdm(loader, leave=True)

    model.train()

    stat = {"dice":  [], "ce": []}

    for i, (x, y) in enumerate(loop):
        #if i >= 20:
        #    break

        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        seg = model(x)
      
        ce_loss = CELoss(seg, y)
        dice_loss = DICELoss(seg, y)
        loss = ce_loss + dice_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 100 == 0:
            loop.set_postfix(
                dice=dice_loss.item(),
                entropy=ce_loss.item(),
                confidance=np.exp(-ce_loss.item()),
            )
            stat["dice"] += [dice_loss.item()]
            stat["ce"] += [ce_loss.item()]
    stat["dice"] = [ mean(stat["dice"]) ]
    stat["ce"] = [ mean(stat["ce"]) ]

    return stat, ce_loss + dice_loss


def main():
    model = UNet(3,6).to(config.DEVICE)

    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')

    CELoss = CrossEntropyLoss()
    #CELoss = nn.CrossEntropyLoss()
    DICELoss = DiceLoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT, model, opt, config.LEARNING_RATE)
    
    train_dataset = YooxDatasetSegmentation()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        # num_workers=config.NUM_WORKERS,
    )
    val_dataset = YooxDatasetSegmentation(True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    stats = {"dice":  [], "ce": []}
    # for epoch in range(config.NUM_EPOCHS):
    for epoch in range(config.NUM_EPOCHS):
        stat, loss = train(model, train_loader, opt, CELoss, DICELoss)
        stats["dice"] += stat["dice"]
        stats["ce"] += stat["ce"]

        path = os.path.dirname(os.path.realpath(__file__))

        fig = plt.figure(figsize=(10,10))
        plt.plot(np.array(stats["dice"]))
        fig.savefig(f'{path}/dice.png')
        plt.close(fig)

        fig = plt.figure(figsize=(10,10))
        plt.plot(np.array(stats["ce"]))
        fig.savefig(f'{path}/ce.png')
        plt.close(fig)

        if config.SAVE_MODEL and epoch % 1 == 0:
            if loss != loss: # not NaN
                load_checkpoint(config.CHECKPOINT, model, opt, config.LEARNING_RATE)
            else:
                save_checkpoint(model, opt, filename=config.CHECKPOINT)
        save_some_examples(model, val_loader)
        loss = eval_some_examples(model, val_loader)

        scheduler.step(loss)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()