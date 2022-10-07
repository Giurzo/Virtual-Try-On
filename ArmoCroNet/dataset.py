from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import cv2
import os

import pandas as pd

from glob import glob

def get_norm_hist(img):
    h0 = torch.histc(img[0], 256)
    h1 = torch.histc(img[1], 256)
    h2 = torch.histc(img[2], 256)
    h = torch.stack([h0,h1,h2])

    h = h.float()
    h /= h.max()
    return h

class YooxDatasetHistogram(Dataset):
    """Yoox Dataset"""
    def __init__(self, get_img=False):
        self.get_img = get_img

        self.df = pd.read_csv("dataset/yu-vton.csv")

        category = self.df["garment"].apply(lambda x: x.split("/")[2])
        self._macros = macros = category.apply(lambda x: x.split("_")[0])
        self._micros = micros = category.apply(lambda x: x.split("_")[1])

        self.macros = list(macros.unique())
        self.micros = list(micros.unique())
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            return self.__getmultipleitem__(idx)

        path0 = self.df.iloc[idx]["target"]
        img = cv2.imread(f"dataset/{path0}", cv2.IMREAD_COLOR)
        img = cv2.resize(img, (192,256))

        img = torch.tensor(img).float()

        h = get_norm_hist(img)

        if self.get_img:
            return h, h, img

        return h, h

    def __getmultipleitem__(self, idx):
        if torch.is_tensor(idx) or isinstance(idx, np.ndarray):
            idx = idx.tolist()
        ret = []
        for i in idx:
            ret.append(self.__getitem__(i))
        return ret


if __name__ == "__main__":
    dataset = YooxDatasetHistogram()
    dataset[0] # get first data

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    item = iter(loader).next()

    print( item[0][0].numpy() )