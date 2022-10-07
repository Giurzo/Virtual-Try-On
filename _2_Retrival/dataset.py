from torch.utils.data import DataLoader

import torch
import cv2
import numpy as np

import os, sys
sys.path.append(os.getcwd())

from __Utils.dataset import YooxTrainTestDataset
from _0_Segmentation.dataset import dense_map

def get_norm_hist(img):
    h0 = torch.histc(img[0], 256)
    h1 = torch.histc(img[1], 256)
    h2 = torch.histc(img[2], 256)
    h = torch.stack([h0,h1,h2])

    h = h.float()
    h /= h.max()
    return h

def get_multi_level_hist(img):
    bins = 8
    hist = torch.zeros((bins,bins,bins))

    binned = torch.div(img, 256//bins, rounding_mode='floor')
    for i in range(bins):
        C0 = binned[...,0] == i
        for j in range(bins):
            C1 = binned[...,1] == j
            for k in range(bins):
                C2 = binned[...,2] == k

                hist[i,j,k] = (C0*C1*C2).sum()

    hist = hist.flatten()

    hist = hist / hist.sum()

    return hist

class YooxDatasetSkinHistogram(YooxTrainTestDataset):

    def get_item(self, idx):

        path0 = self.df.iloc[idx]["target"]
        img = cv2.imread(f"dataset/{path0}", cv2.IMREAD_COLOR)
        imgReal = cv2.resize(img, (192,256))

        pathM = self.df.iloc[idx]["parsing"]
        img = cv2.imread(f"dataset/{pathM}", cv2.IMREAD_COLOR)
        img = cv2.resize(img, (192,256))
        mask = img.transpose((2,0,1))
                
        skin = mask.copy()
        skin = dense_map(skin)
        skin = (skin == 3) +  (skin == 8) +  (skin == 14) +  (skin == 15)
        skin = (skin > 0).flatten()

        flat = imgReal.reshape([192*256, -1])
        skin = flat[skin, :]

        skin = torch.tensor(skin).byte()
        imgReal = torch.tensor(imgReal).float() / 255

        h = get_multi_level_hist(skin)

        return h, imgReal

class YooxDatasetDressHistogram(YooxTrainTestDataset):

    def get_item(self, idx):

        path0 = self.df.iloc[idx]["target"]
        img = cv2.imread(f"dataset/{path0}", cv2.IMREAD_COLOR)
        imgReal = cv2.resize(img, (192,256))

        pathM = self.df.iloc[idx]["parsing"]
        img = cv2.imread(f"dataset/{pathM}", cv2.IMREAD_COLOR)
        img = cv2.resize(img, (192,256))
        mask = img.transpose((2,0,1))
                
        skin = mask.copy()
        skin = dense_map(skin)
        skin = (skin == 1) +  (skin == 2) +  (skin == 4) +  (skin == 7) +  (skin == 12)
        skin = (skin > 0).flatten()

        flat = imgReal.reshape([192*256, -1])
        skin = flat[skin, :]

        skin = torch.tensor(skin).byte()
        imgReal = torch.tensor(imgReal).float() / 255

        h = get_multi_level_hist(skin)

        return h, imgReal


if __name__ == "__main__":
    dataset = YooxDatasetSkinHistogram()
    dataset[0] # get first data

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    item = iter(loader).next()

    print( item[0][0].numpy() )
    print( item[0][0].shape )