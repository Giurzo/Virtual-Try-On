import os
import random
import numpy as np
import torch
import config
from tqdm import tqdm
import cv2
import time

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
        os.mkdir(path)


""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    create_dir("files")
    create_dir("files/gan")
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


def save_some_examples(gen, val_loader):
    create_dir("results")

    H = 256
    W = 192
    size = (W, H)
    checkpoint_path = "files/checkpoint.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = gen
    model = model.to(device)
    model.eval()
    
    time_taken = []

    i = 0
    for x, z, y in val_loader:
        x = x.to(device).float()
        z = z.to(device).float()

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(torch.concat([x,z],1))

            # pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            pred_y = pred_y[0].cpu().numpy() * 255 ## (1, 512, 512)
            pred_y = pred_y.transpose(1,2,0)

            y = y[0].numpy() * 255
            y = y.transpose(1,2,0)

            x = x.cpu()[0].numpy().transpose(1,2,0)*255
            z = z.cpu()[0].numpy().transpose(1,2,0)*255
            #pred_y = pred_y * 255.0
            #pred_y = np.array(pred_y, dtype=np.uint8)

        res = np.concatenate([z, x, pred_y, y], axis=1)

        cv2.imwrite(f"results/{i}.png", res.astype(np.uint8))
        i += 1

        if i > 50:
            break

    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)