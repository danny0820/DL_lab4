"""
TransUNet 訓練腳本
用於訓練 TransUNet 模型進行乳腺癌語義分割
直接適配 DL_lab4 的訓練框架
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import create_df, BCSSDataset
from transunet import TransUNet
from train_transunet import fit_transunet
from metrics import DiceLoss
from utils import plot_loss, plot_score, plot_acc
import torch.nn.functional as F


def main():
    # 設置使用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

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

    # 訓練集數據增強
    transforms_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # 驗證集數據增強
    transforms_val = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # 創建數據集
    train_set = BCSSDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, X_train, mean, std, transforms_train)
    val_set = BCSSDataset(VAL_IMAGE_PATH, VAL_MASK_PATH, X_val, mean, std, transforms_val)

    # 創建數據加載器
    batch_size = 32  # TransUNet 較大，減小 batch size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    # 初始化 TransUNet 模型
    print("\n初始化 TransUNet 模型...")
    model = TransUNet(
        img_ch=3,           # RGB 輸入
        output_ch=3,        # 3類分割
        img_size=512,       # 圖像大小
        hidden_size=768,    # Transformer hidden size
        num_layers=12,      # Transformer layers
        num_heads=12,       # Attention heads
        mlp_dim=3072,       # MLP dimension
        decoder_channels=(256, 128, 64, 16),
        skip_channels=[512, 256, 64, 16],
        n_skip=3,           # 使用 3 個 skip connections
        use_hybrid=True     # 使用 ResNet50 作為 hybrid backbone
    )

    # 計算模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數量: {total_params:,}")
    print(f"可訓練參數量: {trainable_params:,}")

    # 如果有多個GPU，使用 DataParallel 進行多GPU訓練
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 個GPU進行訓練")
        model = nn.DataParallel(model)

    model = model.to(device)
    print(f"模型已移動到: {device}")

    # 訓練超參數（參考 TransUNet 原始設定）
    base_lr = 0.01          # TransUNet 使用 SGD with lr=0.01
    epoch = 150             # TransUNet 原始使用 150 epochs
    weight_decay = 0.0001
    momentum = 0.9          # SGD momentum

    # 損失函數（參考 TransUNet 原始設定）
    # 使用 CrossEntropyLoss + DiceLoss，權重各 0.5
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss()
    
    # 優化器（參考 TransUNet 原始設定）
    # 使用 SGD with momentum=0.9
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=base_lr, 
        momentum=momentum, 
        weight_decay=weight_decay
    )

    print("\n開始訓練...")
    print(f"Batch size: {batch_size}")
    print(f"Total epochs: {epoch}")
    print(f"Base learning rate: {base_lr}")
    print(f"Momentum: {momentum}")
    print(f"Weight decay: {weight_decay}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Max iterations: {epoch * len(train_loader)}")
    print(f"Optimizer: SGD")
    print(f"Scheduler: Poly LR (power=0.9)")
    print(f"Loss: 0.5 * CrossEntropy + 0.5 * Dice")

    # 訓練模型
    history = fit_transunet(
        epoch, 
        model, 
        train_loader, 
        val_loader, 
        criterion_ce,
        criterion_dice,
        optimizer,
        base_lr,
        device,
        loss_weights=(0.5, 0.5)  # TransUNet 原始設定
    )

    # 保存最終模型
    torch.save(model, 'TransUNet_final.pt')
    print("\n模型已保存為 TransUNet_final.pt")

    # 繪製訓練曲線
    print("\n繪製訓練曲線...")
    plot_loss(history)
    plot_score(history)
    plot_acc(history)

    print("\n訓練完成！")


if __name__ == '__main__':
    main()
