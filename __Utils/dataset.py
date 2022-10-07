from torch.utils.data import Dataset
import torch

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

class YooxTrainTestDataset(ABC, Dataset):
    def __init__(self, test=False):
        self.test = test
        self.df = pd.read_csv("dataset/yu-vton.csv")
    
    def __len__(self):
        test_len = len(self.df) // 10 * 2
        residual = len(self.df) - ( 10 * test_len // 2 ) - 8
        test_len += max(0, residual)

        if self.test:
            return test_len
        else:
            return len(self.df) - test_len

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            return self.__getmultipleitem__(idx)

        if idx < 0:
            idx += len(self)
        if idx >= len(self):
            idx %= len(self)

        if self.test:
            idx = 8 + 10 * (idx // 2) + idx % 2
        else:
            idx = idx + idx // 8 * 2

        return self.get_item(idx)
    
    def __getmultipleitem__(self, idx):
        if torch.is_tensor(idx) or isinstance(idx, np.ndarray):
            idx = idx.tolist()
        ret = []
        for i in idx:
            ret.append(self.__getitem__(i))
        return ret

    @abstractmethod
    def get_item(self, idx):
        pass