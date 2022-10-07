from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import cv2
import os

import pandas as pd

from glob import glob

class YooxDatasetEdge(Dataset):
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

        x = cv2.Canny(img, 50, 200, None, 3)
        y = img.transpose((2,0,1))

        return x/255, y/255

    def __getmultipleitem__(self, idx):
        if torch.is_tensor(idx) or isinstance(idx, np.ndarray):
            idx = idx.tolist()
        ret = []
        for i in idx:
            ret.append(self.__getitem__(i))
        return ret


if __name__ == "__main__":
    dataset = YooxDatasetEdge()
    dataset[0] # get first data

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    item = iter(loader).next()

    cv2.imshow("0", item[0][0].numpy())
    cv2.imshow("1", item[1][0].numpy().transpose((1,2,0)))
    cv2.waitKey()
    cv2.destroyAllWindows()