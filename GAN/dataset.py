from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import cv2
import os

import pandas as pd

from glob import glob

class YooxDatasetPairs(Dataset):
    """Yoox Dataset"""
    def __init__(self):

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

        command = self.df.iloc[idx]["garment"][0]

        path0 = self.df.iloc[idx]["garment"]
        img = cv2.imread(f"dataset/{path0}", cv2.IMREAD_COLOR)
        img = cv2.resize(img, (192,256))
        img0 = img.transpose((2,0,1))

        path1 = self.df.iloc[idx]["target"]
        img = cv2.imread(f"dataset/{path1}", cv2.IMREAD_COLOR)
        img = cv2.resize(img, (192,256))
        img1 = img.transpose((2,0,1))

        pathM = self.df.iloc[idx]["parsing"]
        img = cv2.imread(f"dataset/{pathM}", cv2.IMREAD_COLOR)
        img = cv2.resize(img, (192,256))
        mask = img.transpose((2,0,1))
        
        if command == "u":
            mask = torch.eq(torch.tensor(mask), torch.tensor([[[128]],[[0]],[[0]]])).all(0)
        if command == "l":
            mask = torch.eq(torch.tensor(mask), torch.tensor([[[128]],[[128]],[[0]]])).all(0)

        mask = img1 - img1 * mask.numpy() # .astype(np.uint8)*255
        return img0/255, mask/255, img1/255 # self.macros.index(self._macros[idx]), self.micros.index(self._micros[idx])
    
    def __getmultipleitem__(self, idx):
        if torch.is_tensor(idx) or isinstance(idx, np.ndarray):
            idx = idx.tolist()
        ret = []
        for i in idx:
            ret.append(self.__getitem__(i))
        return ret


if __name__ == "__main__":
    dataset = YooxDatasetPairs()
    dataset[0] # get first data
    dataset[0][2] # get only label
    dataset[[0,1]] # get multiple data

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    item = iter(loader).next()

    cv2.imshow("0", item[0][0].numpy().transpose((1,2,0)))
    cv2.imshow("1", item[1][0].numpy().transpose((1,2,0)))
    cv2.imshow("2", item[2][0].numpy().transpose((1,2,0)))
    cv2.waitKey()
    cv2.destroyAllWindows()