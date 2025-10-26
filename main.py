"""
主訓練腳本
用於訓練 Attention U-Net 模型進行乳腺癌語義分割
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import create_df, BCSSDataset
from model import AttentionUNet
# from unet3plus import UNet3Plus
from metrics import DiceLoss, FocalLoss
from train import fit
from utils import plot_loss, plot_score, plot_acc


def main():
    # 設置使用GPU 1和2
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

    # 檢查可用的GPU數量
    if torch.cuda.is_available():
        print(f"可用的GPU數量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("使用CPU訓練")

    print(f"主要設備: {device}")

    # 數據路徑
    TRAIN_IMAGE_PATH = './BCSS/train/'
    VAL_IMAGE_PATH = './BCSS/val/'
    TRAIN_MASK_PATH = './BCSS/train_mask/'
    VAL_MASK_PATH = './BCSS/val_mask/'

    # 創建數據框
    train_df = create_df(TRAIN_IMAGE_PATH)
    val_df = create_df(VAL_IMAGE_PATH)

    print('Total Train Images: ', len(train_df))
    print('Total Val Images: ', len(val_df))

    X_train = train_df['id'].to_numpy()
    X_val = val_df['id'].to_numpy()

    # 數據增強設置
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 訓練集數據增強 - 使用 albumentations
    transforms_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # 驗證集數據增強 - 使用 albumentations
    transforms_val = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # 創建數據集
    train_set = BCSSDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, X_train, mean, std, transforms_train)
    val_set = BCSSDataset(VAL_IMAGE_PATH, VAL_MASK_PATH, X_val, mean, std, transforms_val)

    # 創建數據加載器
    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # 初始化模型
    model = AttentionUNet(3, 3)

    # 如果有多個GPU，使用 DataParallel 進行多GPU訓練
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 個GPU進行訓練")
        model = nn.DataParallel(model)

    model = model.to(device)
    print(f"模型已移動到: {device}")

    # 訓練超參數
    max_lr = 0.001
    epoch = 100
    weight_decay = 0.0001

    # 損失函數和優化器
    # 使用 Focal Loss 替代 CrossEntropyLoss
    # gamma=2.0 是標準設置，alpha 可以根據類別不平衡情況調整
    criterion1 = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
    criterion2 = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch, steps_per_epoch=len(train_loader))

    # 訓練模型
    history = fit(
        epoch, 
        model, 
        train_loader, 
        val_loader, 
        criterion1, 
        criterion2, 
        optimizer, 
        scheduler,
        device
    )

    # 保存最終模型
    torch.save(model, 'AttentionUNet.pt')
    print("模型已保存為 AttentionUNet.pt")

    # 繪製訓練曲線
    print("\n繪製訓練曲線...")
    plot_loss(history)
    plot_score(history)
    plot_acc(history)


if __name__ == '__main__':
    main()
