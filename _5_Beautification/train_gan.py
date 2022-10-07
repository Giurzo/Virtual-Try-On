import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

import os, sys
sys.path.append(os.getcwd())

from utils import save_checkpoint, load_checkpoint, save_some_examples
import config

from _5_Beautification.model_generator import UNet
from _0_Segmentation.model_discriminator import Discriminator

from tqdm import tqdm
from _4_TPS.dataset import YooxDatasetPairs

def train(fitter, gen, dis, loader, opt_gen, opt_dis, CELoss, DICELoss, BCELoss):
    loop = tqdm(loader, leave=True)

    fitter.eval()
    gen.train()
    dis.train()

    for i, inputs in enumerate(loop):
        if i > 500:
            break

        im = inputs['image'].cuda().float()
        p = inputs['person'].cuda().float()
        img_mask = inputs['img_mask'].cuda().float()
        c = inputs['cloth'].cuda().float()
        cm = inputs['cloth_mask'].cuda().float()
        im_g = inputs['grid_image'].cuda().float()
        mask = inputs['only_cloth'].cuda().float()
        skin = inputs['skin'].cuda().float()

        with torch.no_grad():
            grid, theta = fitter(cm.float(), img_mask.float())
            warped_cloth = F.grid_sample(c, grid, padding_mode='border')
            warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')

        
        warped_cloth.requires_grad = True
        warped_mask.requires_grad = True
        y_fake = gen(torch.cat([warped_cloth, warped_mask, mask, img_mask, skin],1))

        # Train Discriminator
        D_fake = dis(y_fake.detach())
        D_fake_loss = BCELoss(D_fake, torch.zeros_like(D_fake))
        D_real = dis(p.detach())
        D_real_loss = BCELoss(D_real, torch.ones_like(D_real))
        D_loss = D_real_loss + D_fake_loss

        opt_dis.zero_grad()
        D_loss.backward()
        opt_dis.step()
      
        # Train Generator
        D_fake = dis(y_fake)
        G_fake_loss = BCELoss(D_fake, torch.ones_like(D_fake))

        ce_loss1 = CELoss(y_fake, p)
        ce_loss2 = CELoss(y_fake, p * (1 - warped_mask * (1 - skin)) + warped_cloth * warped_mask * (1 - skin))
        #dice_loss = DICELoss(seg, y)
        loss = G_fake_loss + (ce_loss1 + ce_loss2) * 10**(i%2)

        opt_gen.zero_grad()
        loss.backward()
        opt_gen.step()

        if i % 7 == 0:
            loop.set_postfix(
                MSE1=ce_loss1.item(),
                MSE2=ce_loss2.item(),
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                G_fake=G_fake_loss.item(),
            )
    return G_fake_loss + D_loss

from _4_TPS.train import get_opt, load_checkpoint as LOAD
from _4_TPS.networks import GMM
def main():
    gen = UNet(9,6,3).to(config.DEVICE)
    dis = Discriminator(in_channels=3).to(config.DEVICE)

    opt = get_opt()
    opt.checkpoint = "checkpoints/GMM/TPS_best.pth"
    fitter = GMM(opt).to(config.DEVICE)
    if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
        LOAD(fitter, opt.checkpoint)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE_GEN, betas=(0.5, 0.999))
    opt_dis = optim.Adam(dis.parameters(), lr=config.LEARNING_RATE_DIS, betas=(0.5, 0.999))
    
    #CELoss = CrossEntropyLoss()
    CELoss = nn.MSELoss()
    BCELoss = nn.BCEWithLogitsLoss()


    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE_GEN)
        load_checkpoint(config.CHECKPOINT_DIS, dis, opt_dis, config.LEARNING_RATE_DIS)
    
    train_dataset = YooxDatasetPairs()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        # num_workers=config.NUM_WORKERS,
    )
    val_dataset = YooxDatasetPairs()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # for epoch in range(config.NUM_EPOCHS):
    for epoch in range(config.NUM_EPOCHS):
        loss = train(fitter, gen, dis, train_loader, opt_gen, opt_dis, CELoss, CELoss, BCELoss)

        if config.SAVE_MODEL and epoch % 1 == 0:
            if loss != loss: # non NaN
                load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE_GEN)
                load_checkpoint(config.CHECKPOINT_DIS, gen, opt_dis, config.LEARNING_RATE_DIS)
            else:
                # if config.LOAD_MODEL == False:
                #     print("#"*20)
                #     risposta = input("Vuoi sovrascrivere perdendo il modello precedente? SI - NO:\n")
                #     print("#"*20)
                #     if risposta == "SI":
                #         save_checkpoint(model, opt, filename=config.CHECKPOINT)
                #         config.LOAD_MODEL = True
                # else:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(dis, opt_dis, filename=config.CHECKPOINT_DIS)
        save_some_examples(fitter, gen, val_loader)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()