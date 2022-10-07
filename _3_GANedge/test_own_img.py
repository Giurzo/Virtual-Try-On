import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from utils import load_checkpoint
import config

from model_generator import Generator

import os, sys
sys.path.append(os.getcwd())

from tqdm import tqdm


def main():
    model = Generator(in_channels=1,levels=6).to(config.DEVICE)

    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    
    load_checkpoint(config.CHECKPOINT_GEN, model, opt, config.LEARNING_RATE)
    

    import cv2
    img = cv2.imread("_3_GANedge/Immagine.png", cv2.IMREAD_GRAYSCALE)
    model.eval()

    y = torch.tensor(cv2.imread("_3_GANedge/maglia.jpg").transpose(2,0,1)).to(config.DEVICE).float()[None]
    h = None
    for b in y:
        h0 = torch.histc(b[0], 256)
        h1 = torch.histc(b[1], 256)
        h2 = torch.histc(b[2], 256)
        h_3c = torch.stack([h0,h1,h2])
        if h is None:
            h = h_3c[None,:]
        else:
            h = torch.concat([h, h_3c[None,:]], 0)
    h = h.to(config.DEVICE).float()

    y = model(torch.tensor(img)[None,None].float().cuda()/255, h)

    

    cv2.imwrite("_3_GANedge/drawn.png", (y[0].detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()