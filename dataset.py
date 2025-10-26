import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import os


def create_df(IMAGE_PATH):
    """創建包含圖像ID的DataFrame"""
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))


class BCSSDataset(Dataset):
    """BCSS數據集類別，用於訓練和驗證"""
    
    def __init__(self, img_path, mask_path, X, mean, std, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        # 如果有變換（使用 albumentations）
        if self.transform is not None:
            # albumentations 需要使用命名參數
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            # ToTensorV2() 已經將 mask 轉換為 tensor，直接轉換為 long 類型
            mask = mask.long()
        else:
            # 如果沒有變換，手動進行標準化（使用 torchvision）
            from torchvision import transforms as T
            t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
            img = t(img)
            # mask 轉換為 tensor
            mask = torch.from_numpy(mask).long()

        return img, mask


class BCSSTestDataset(Dataset):
    """BCSS測試數據集類別"""
    def __init__(self, img_path, X, transform=None):
        self.img_path = img_path
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            aug = self.transform(image=img)
            img = Image.fromarray(aug['image'])

        if self.transform is None:
            img = Image.fromarray(img)

        return img
