import os
import random
import numpy as np
import torch
import config
from tqdm import tqdm
import cv2
import time

import torch.nn.functional as F

""" Seeding the randomness """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_checkpoint(model, optimizer, filename):
    create_dir(os.path.dirname(filename))
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    optimizer.param_groups[0]['capturable'] = True

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_some_examples(fitter, model, val_loader):
    create_dir("results/beauty")

    size = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)

    model = model.to(config.DEVICE)
    model.eval()
    
    time_taken = []

    for idx, inputs in enumerate(val_loader):
        
        im = inputs['image'].cuda().float()
        p = inputs['person'].cuda().float()
        img_mask = inputs['img_mask'].cuda().float()
        c = inputs['cloth'].cuda().float()
        cm = inputs['cloth_mask'].cuda().float()
        im_g = inputs['grid_image'].cuda().float()
        mask = inputs['only_cloth'].cuda().float()
        skin = inputs['skin'].cuda().float()


        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()

            grid, theta = fitter(cm.float(), img_mask.float())
            warped_cloth = F.grid_sample(c, grid, padding_mode='border')
            warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')

            y_fake = model(torch.concat([warped_cloth, warped_mask, mask, img_mask, skin],1))

            total_time = time.time() - start_time
            time_taken.append(total_time)

        y_fake = y_fake[0].cpu().numpy().transpose(1,2,0)

        y = p[0].cpu().numpy().transpose(1,2,0)
        x = (mask + warped_cloth * warped_mask * img_mask)[0].cpu().numpy().transpose(1,2,0)
        x = ( p * (1 - warped_mask * (1 - skin)) + warped_cloth * warped_mask * (1 - skin) )[0].cpu().numpy().transpose(1,2,0)
        q = (mask + warped_cloth * warped_mask)[0].cpu().numpy().transpose(1,2,0)
        warped_cloth = warped_cloth[0].cpu().numpy().transpose(1,2,0)
        res = np.concatenate([q, warped_cloth, x, y_fake, y], axis=1)

        cv2.imwrite(f"results/beauty/{idx}.png", (res*255).astype(np.uint8))

        if idx > 50:
            break

    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)