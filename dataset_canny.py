from torch.utils.data import Dataset, DataLoader, random_split
import torch
import cv2
import os
import numpy as np

class YooxDataset(Dataset):
    """Yoox Dataset"""

    def __init__(self):
        self.macros = []
        self.micros = []
        self.df = []

        for foldername in os.listdir("dataset/lower_body/images"):
            categories = foldername.split("_")
            self.macros.append(categories[0])
            self.micros.append(categories[1])
            for filename in os.listdir("dataset/lower_body/images/"+foldername):
                tup = ("dataset/lower_body/images/"+foldername+"/"+filename, categories[0], categories[1])
                self.df.append(tup)

        for foldername in os.listdir("dataset/upper_body/images"):
            categories = foldername.split("_")
            self.macros.append(categories[0])
            self.micros.append(categories[1])
            for filename in os.listdir("dataset/upper_body/images/"+foldername):
                tup = ("dataset/upper_body/images/"+foldername+"/"+filename, categories[0], categories[1])
                self.df.append(tup)

        self.macros = list(set(self.macros))
        self.micros = list(set(self.micros))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            return self.__getmultipleitem__(idx)

        img = cv2.imread(self.df[idx][0], cv2.IMREAD_COLOR)
        img = cv2.resize(img, [192,256])
        can = cv2.Canny(img, 50, 200, None, 3)
        img = np.concatenate((img, can[:,:,None]), axis=2)
        img = np.transpose(img, (2, 0, 1))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = (cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)**2 + cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)**2)**0.5
        return img, self.macros.index(self.df[idx][1])#, self.df[idx][1]
    
    def __getmultipleitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ret = []
        for i in idx:
            ret.append(self.__getitem__(i))
        return ret

dataset = YooxDataset()

train_len = len(dataset) * 8 // 10
test_len = len(dataset) - train_len
train_dataset, test_dataset = random_split(dataset,[train_len,test_len])

#print(dataset[0])
#print(dataset[[0,1]])
print(len(train_dataset))
print(len(test_dataset))

loader = DataLoader(dataset, batch_size=64)
#print(iter(loader).next())

#cv2.imshow("dst", train_dataset[0][0])

k = cv2.waitKey(0)

#print(train_dataset[0][1])

print(len(dataset.macros))