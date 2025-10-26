"""
主訓練腳本 - Deep Supervision 版本
用於訓練 UNet3Plus with Deep Supervision 模型進行乳腺癌語義分割
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import create_df, BCSSDataset
# from unet3plus_deep_sup import UNet3PlusDeepSup
from attention_unet_deep_sup import AttentionUNetDeepSup
from metrics import DiceLoss, FocalLoss
from train_deepsup import fit_deepsup
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

    # ==================== 初始化 Deep Supervision 模型 ====================
    print("\n" + "=" * 70)
    print("初始化 UNet3Plus with Deep Supervision 模型")
    print("=" * 70)
    
    model = AttentionUNetDeepSup(img_ch=3, output_ch=3)
    model_name = 'AttentionUNetDeepSup'

    # 如果有多個GPU，使用 DataParallel 進行多GPU訓練
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 個GPU進行訓練")
        model = nn.DataParallel(model)

    model = model.to(device)
    print(f"模型已移動到: {device}")

    # 訓練超參數
    max_lr = 0.001
    epoch = 100
    weight_decay = 0.003

    print("\n訓練超參數:")
    print(f"  - 最大學習率: {max_lr}")
    print(f"  - 訓練輪數: {epoch}")
    print(f"  - 權重衰減: {weight_decay}")
    print(f"  - Batch Size: {batch_size}")

    # 損失函數和優化器
    # 使用 Focal Loss 替代 CrossEntropyLoss
    # gamma=2.0 是標準設置，alpha 可以根據類別不平衡情況調整
    criterion1 = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
    criterion2 = DiceLoss()
    
    print("\n損失函數:")
    print(f"  - Criterion 1: Focal Loss (alpha=1.0, gamma=2.0)")
    print(f"  - Criterion 2: Dice Loss")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,                         # 你要爬到的峰值 LR
        epochs=epoch,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,                        # ← 前 10% 的步數做 warm-up（可改 0.03~0.1）
        anneal_strategy='cos',                # ← 後段用 cosine 衰減
        div_factor=25.0,                      # ← 初始 LR = max_lr / 25  (例如 1e-3/25 ≈ 4e-5)
        final_div_factor=1e4,                 # ← 最終 LR ≈ 初始 LR / 1e4 -> 幾乎降到 0
        three_phase=False
    )

    print(f"\n優化器: AdamW")
    print(f"學習率調度器: OneCycleLR")

    # 訓練模型 - 使用 Deep Supervision 版本的訓練函數
    print("\n" + "=" * 70)
    print("開始訓練...")
    print("=" * 70 + "\n")
    
    history = fit_deepsup(
        epoch, 
        model, 
        train_loader, 
        val_loader, 
        criterion1, 
        criterion2, 
        optimizer, 
        scheduler,
        device,
        model_name=model_name
    )

    # 保存最終模型
    final_model_path = f'{model_name}_final.pt'
    torch.save(model, final_model_path)
    print(f"\n最終模型已保存為: {final_model_path}")

    # 繪製訓練曲線
    print("\n繪製訓練曲線...")
    plot_loss(history)
    plot_score(history)
    plot_acc(history)
    
    print("\n" + "=" * 70)
    print("✅ 訓練完成！")
    print("=" * 70)
    print(f"\n保存的模型檔案:")
    print(f"  - 最佳損失模型: {model_name}_best_loss.pt")
    print(f"  - 最佳 mIoU 模型: {model_name}_best_miou.pt")
    print(f"  - 最終模型: {final_model_path}")


if __name__ == '__main__':
    main()
