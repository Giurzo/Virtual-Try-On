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
from model_discriminator import Discriminator

import os, sys
sys.path.append(os.getcwd())
from __Utils.DiceLoss import DiceLoss
from __Utils.CrossEntropyLoss import CrossEntropyLoss

from tqdm import tqdm


def train(gen, dis, loader, opt_gen, opt_dis, CELoss, DICELoss, BCELoss):
    loop = tqdm(loader, leave=True)

    gen.train()
    dis.train()

    stat = {"dice":  [], "ce": []}

    for i, (x, y) in enumerate(loop):
        if i > 1000:
            break

        opt_dis.zero_grad()
        opt_gen.zero_grad()

        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        seg = gen(x)

        seg_idx = seg.argmax(1)
        seg_onehot = torch.eye(seg.shape[1])[seg_idx].permute(0,3,1,2).float().to(config.DEVICE)
        y_onehot = torch.eye(seg.shape[1])[y].permute(0,3,1,2).float().to(config.DEVICE)


        # Train Discriminator
        D_fake = dis(seg_onehot.detach())
        D_fake_loss = BCELoss(D_fake, torch.zeros_like(D_fake))
        D_real = dis(y_onehot.detach())
        D_real_loss = BCELoss(D_real, torch.ones_like(D_real))
        D_loss = D_real_loss + D_fake_loss

        D_loss.backward()
        opt_dis.step()
      
        # Train Generator
        D_fake = dis(seg_onehot)
        G_fake_loss = BCELoss(D_fake, torch.ones_like(D_fake))

        ce_loss = CELoss(seg, y)
        dice_loss = DICELoss(seg, y)
        loss = ce_loss + dice_loss + G_fake_loss

        loss.backward()
        opt_gen.step()

        if i % 100 == 0:
            loop.set_postfix(
                dice=dice_loss.item(),
                entropy=ce_loss.item(),
                confidance=np.exp(-ce_loss.item()),
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                G_fake=G_fake_loss.item(),
            )
            stat["dice"] += [dice_loss.item()]
            stat["ce"] += [ce_loss.item()]
    stat["dice"] = [ mean(stat["dice"]) ]
    stat["ce"] = [ mean(stat["ce"]) ]

    return stat, ce_loss + dice_loss + G_fake_loss + D_loss


def main():
    gen = UNet(3,6).to(config.DEVICE)
    dis = Discriminator(in_channels=16).to(config.DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE_GEN, betas=(0.5, 0.999))
    opt_dis = optim.Adam(dis.parameters(), lr=config.LEARNING_RATE_DIS, betas=(0.5, 0.999))
    scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(opt_gen, 'min')
    scheduler_dis = optim.lr_scheduler.ReduceLROnPlateau(opt_dis, 'min')

    CELoss = CrossEntropyLoss()
    #CELoss = nn.CrossEntropyLoss()
    DICELoss = DiceLoss()
    BCELoss = nn.BCEWithLogitsLoss()


    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE_GEN)
        load_checkpoint(config.CHECKPOINT_DIS, dis, opt_dis, config.LEARNING_RATE_DIS)
    
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
        stat, loss = train(gen, dis, train_loader, opt_gen, opt_dis, CELoss, DICELoss, BCELoss)

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

        if config.SAVE_MODEL:
            if loss != loss: # non NaN
                load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE_GEN)
                load_checkpoint(config.CHECKPOINT_DIS, gen, opt_dis, config.LEARNING_RATE_DIS)
            else:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(dis, opt_dis, filename=config.CHECKPOINT_DIS)
        save_some_examples(gen, val_loader)
        loss = eval_some_examples(gen, val_loader)

        scheduler_gen.step(loss)
        scheduler_dis.step(loss)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()