from pickletools import uint8
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import cv2

import pandas as pd

from glob import glob

import os, sys
sys.path.append(os.getcwd())
from _0_Segmentation.dataset import dense_map

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

        # mode =  None
        mode =  cv2.IMREAD_COLOR

        path0 = self.df.iloc[idx]["garment"]
        img = cv2.imread(f"dataset/{path0}", mode)
        img = cv2.resize(img, (192,256))
        img0 = img.transpose((2,0,1))

        path1 = self.df.iloc[idx]["target"]
        img = cv2.imread(f"dataset/{path1}", mode)
        img = cv2.resize(img, (192,256))
        img1 = img.transpose((2,0,1))

        pathM = self.df.iloc[idx]["parsing"]
        img = cv2.imread(f"dataset/{pathM}", mode)
        img = cv2.resize(img, (192,256))
        mask = img.transpose((2,0,1))

        # Mask converted from uint8 -> float64 in order to make it printable 
        cloth_mask = self.to_mask(img0.transpose((1,2,0)))
        cloth_mask = cloth_mask.reshape((256,192,1)).transpose((2,0,1))
        
        skin = mask.copy()
        skin = dense_map(skin)
        skin = (skin == 3) +  (skin == 8) +  (skin == 14) +  (skin == 15)
        skin = skin[None,:] > 0

        if command == "u":
            mask = torch.eq(torch.tensor(mask), torch.tensor([[[128]],[[0]],[[0]]])).all(0)
        if command == "l":
            mask = torch.eq(torch.tensor(mask), torch.tensor([[[128]],[[128]],[[0]]])).all(0)

        
        r_mask = mask.to(torch.uint8).reshape((256,192,1)).numpy().transpose((2,0,1))

        # print(r_mask.shape)
        # print(r_mask.dtype)
        mask = img1 - img1 * mask.numpy() # .astype(np.uint8)*255

        
        grid = cv2.imread(f"_4_TPS/grid.png", mode)
        grid = cv2.resize(grid, (192,256))
        grid = grid.transpose((2,0,1))
        

        result = {
            # 'c_name':   c_name,     # for visualization
            # 'im_name':  im_name,    # for visualization or ground truth
            'cloth':    img0/255,          # for input
            'person':    img1/255,          # for input
            'cloth_mask':     cloth_mask,   # for input
            'image':    img.transpose((2,0,1))/255,         # for visualization
            # 'agnostic': agnostic,   # for input
            'img_mask': r_mask,    # for ground truth
            'only_cloth': mask/255,         # for visualization
            # 'head': im_h,           # for visualization
            # 'pose_image': im_pose,  # for visualization
            'grid_image': grid,     # for visualization
            'skin': skin,
            }
        return result # self.macros.index(self._macros[idx]), self.micros.index(self._micros[idx])
    
    def __getmultipleitem__(self, idx):
        if torch.is_tensor(idx) or isinstance(idx, np.ndarray):
            idx = idx.tolist()
        ret = []
        for i in idx:
            ret.append(self.__getitem__(i))
        return ret

    def to_mask(self, img):
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S

        img = img.astype('uint8')
        # Output dtype = cv2.CV_8U
        img = cv2.GaussianBlur(img, (3, 3), 0)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        # Gradient-Y
        # grad_y = cv.Scharr(gray,ddepth,0,1)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        cl = clahe.apply(grad)

        otsu_threshold, image_result = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)

        FG = np.ones_like(image_result)

        BG = image_result.cumsum(0) == 0
        FG[BG] = 0

        BG = image_result[::-1].cumsum(0) == 0
        FG[BG[::-1]] = 0

        BG = image_result.cumsum(1) == 0
        FG[BG] = 0

        BG = image_result[:,::-1].cumsum(1) == 0
        FG[BG[:,::-1]] = 0

        return FG

class YooxDataLoader(object):
    def __init__(self, opt, dataset):
        super(YooxDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    dataset = YooxDatasetPairs()
    # dataset[0] # get first data
    # dataset[0][2] # get only label
    # dataset[[0,1]] # get multiple data

    # loader = DataLoader(dataset, batch_size=64, shuffle=True)
    # item = iter(loader).next()
    # for i in dataset:
        # cv2.imshow("qq", i[3].transpose((1,2,0)))
        # cv2.imshow("0", i[0].transpose((1,2,0)))
        # cv2.imshow("1", item[1][0].numpy().transpose((1,2,0)))
        # cv2.imshow("2", item[2][0].numpy().transpose((1,2,0)))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    print([(k,v.shape, v.dtype) for k,v in dataset[0].items()])

    # cv2.imshow("0", dataset[0]['cloth'].transpose((1,2,0)))
    # cv2.imshow("1", dataset[0]['cloth_mask'].transpose((1,2,0)))
    # # cv2.imshow("2", dataset[0][3].transpose((1,2,0)))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
