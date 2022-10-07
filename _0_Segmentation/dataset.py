import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader

from albumentations.augmentations.crops.transforms import CenterCrop
from albumentations.augmentations.geometric.resize import RandomScale
from albumentations.augmentations.transforms import GaussNoise
from albumentations.core.composition import Compose

import os, sys
sys.path.append(os.getcwd())

from _0_Segmentation import config
from __Utils.dataset import YooxTrainTestDataset


label_map = {
    "background": 0,
    "upper_clothes": 1,
    "bag": 2,
    "hair": 3,
    "pants": 4,
    "left_leg": 6,
    "right_shoe": 7,
    "left_arm": 8,
    "left_shoe": 12,
    "right_leg": 13,
    "head": 14,
    "right_arm": 15,
    
    "hat": -1,
    "sunglasses": -2,
    "skirt": -3,
    "dress": -4,
    "belt": -5,
    "scarf": -6,
}

def dense_map(img):
    img = img.astype(np.longlong)
    img = img[0] + img[1] * 256 + img[2] * 256 * 256

    old_idx = [0, 128, 16384, 32768, 32896, 4194304, 4194432, 4227072, 4227200, 8388608, 8404992, 8421376, 12582912, 12583040, 12615680, 12615808]
    for i in range(len(old_idx)):
        img[img == old_idx[i]] = i
    return img

class YooxDatasetSegmentation(YooxTrainTestDataset):
    
    def get_item(self, idx):

        path = self.df.iloc[idx]["target"]
        #path = self.df.iloc[idx]["garment"]
        x = cv2.imread(f"dataset/{path}", cv2.IMREAD_COLOR)
        #x = cv2.resize(x, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

        path = self.df.iloc[idx]["parsing"]
        y = cv2.imread(f"dataset/{path}", cv2.IMREAD_COLOR)
        #y = cv2.resize(y, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        y = y.transpose((2,0,1)) # channel first

        y = dense_map(y)

        T_scale = RandomScale((-0.75,-0.5), interpolation=cv2.INTER_NEAREST, always_apply=True, p=1)
        T_crop = CenterCrop(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, always_apply=True, p=1)
        T_noise = GaussNoise(var_limit=(0, 100), mean=0, per_channel=True, p=1 if not self.test else 0)
        T = Compose([T_scale,T_crop,T_noise])

        res = T(image=x, mask=y)
        x = res["image"]
        y = res["mask"]


        x = x.transpose((2,0,1)) # channel first

        return torch.tensor(x).float() / 255, torch.tensor(y).long() # self.macros.index(self._macros[idx]), self.micros.index(self._micros[idx])


if __name__ == "__main__":
    dataset = YooxDatasetSegmentation(True)
    dataset[-1] # get last data
    dataset[1000000000] # get last data
    dataset[0][1] # get only segmentation
    dataset[[0,1]] # get multiple data

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    item = iter(loader).next()

    cv2.imshow("0", item[0][0].numpy().transpose((1,2,0)))
    cv2.imshow("1", (item[1][0].numpy() == 4).astype(np.uint8)*255)
    cv2.waitKey()
    cv2.destroyAllWindows()