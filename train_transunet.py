"""
TransUNet 專用訓練函數
使用與原始 TransUNet 相同的訓練設定
"""

import torch
import numpy as np
import time
from tqdm import tqdm
from metrics import pixel_accuracy, mIoU


def get_lr(optimizer):
    """獲取當前學習率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def poly_lr_scheduler(optimizer, iter_num, max_iterations, base_lr, power=0.9):
    """
    Poly learning rate policy
    使用與原始 TransUNet 相同的學習率調度策略
    lr = base_lr * (1 - iter/max_iter)^power
    """
    lr = base_lr * (1.0 - iter_num / max_iterations) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def fit_transunet(epochs, model, train_loader, val_loader, criterion_ce, criterion_dice, 
                  optimizer, base_lr, device, loss_weights=(0.5, 0.5)):
    """
    TransUNet 訓練函數
    使用與原始 TransUNet 相同的訓練流程和損失函數設定
    
    Args:
        epochs: 訓練輪數
        model: TransUNet 模型
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器
        criterion_ce: CrossEntropyLoss
        criterion_dice: DiceLoss
        optimizer: 優化器 (SGD with momentum)
        base_lr: 基礎學習率
        device: 訓練設備
        loss_weights: CE 和 Dice loss 的權重，預設 (0.5, 0.5)
    
    Returns:
        history: 包含訓練歷史的字典
    """
    torch.cuda.empty_cache()
    
    # 歷史記錄
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    
    # 記錄最佳性能
    min_loss = np.inf
    best_iou = 0.0
    not_improve = 0
    
    # 計算總迭代次數
    max_iterations = epochs * len(train_loader)
    iter_num = 0

    model.to(device)
    fit_time = time.time()
    
    print(f"\n開始訓練 TransUNet...")
    print(f"總 epochs: {epochs}")
    print(f"每個 epoch 迭代次數: {len(train_loader)}")
    print(f"總迭代次數: {max_iterations}")
    print(f"基礎學習率: {base_lr}")
    print(f"損失權重: CE={loss_weights[0]}, Dice={loss_weights[1]}")
    
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        running_loss_ce = 0
        running_loss_dice = 0
        iou_score = 0
        accuracy = 0
        
        # 訓練階段
        model.train()
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {e+1}/{epochs}")):
            image, mask = data

            # 將數據移到指定設備
            image = image.to(device)
            mask = mask.to(device)
            
            # 前向傳播
            output = model(image)
            
            # 計算損失 (0.5 * CrossEntropy + 0.5 * Dice)
            loss_ce = criterion_ce(output, mask)
            loss_dice = criterion_dice(output, mask)
            loss = loss_weights[0] * loss_ce + loss_weights[1] * loss_dice
            
            # 計算指標
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新學習率 (Poly LR policy)
            lr = poly_lr_scheduler(optimizer, iter_num, max_iterations, base_lr, power=0.9)
            lrs.append(lr)
            iter_num += 1

            # 累積損失
            running_loss += loss.item()
            running_loss_ce += loss_ce.item()
            running_loss_dice += loss_dice.item()

        # 驗證階段
        model.eval()
        test_loss = 0
        test_loss_ce = 0
        test_loss_dice = 0
        test_accuracy = 0
        val_iou_score = 0
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader, desc="Validating")):
                image, mask = data

                # 將數據移到指定設備
                image = image.to(device)
                mask = mask.to(device)
                
                # 前向傳播
                output = model(image)
                
                # 計算指標
                val_iou_score += mIoU(output, mask)
                test_accuracy += pixel_accuracy(output, mask)
                
                # 計算損失
                loss_ce = criterion_ce(output, mask)
                loss_dice = criterion_dice(output, mask)
                loss = loss_weights[0] * loss_ce + loss_weights[1] * loss_dice
                
                test_loss += loss.item()
                test_loss_ce += loss_ce.item()
                test_loss_dice += loss_dice.item()

        # 計算平均值
        avg_train_loss = running_loss / len(train_loader)
        avg_train_loss_ce = running_loss_ce / len(train_loader)
        avg_train_loss_dice = running_loss_dice / len(train_loader)
        avg_val_loss = test_loss / len(val_loader)
        avg_val_loss_ce = test_loss_ce / len(val_loader)
        avg_val_loss_dice = test_loss_dice / len(val_loader)
        avg_train_iou = iou_score / len(train_loader)
        avg_val_iou = val_iou_score / len(val_loader)
        avg_train_acc = accuracy / len(train_loader)
        avg_val_acc = test_accuracy / len(val_loader)
        
        # 記錄歷史
        train_losses.append(avg_train_loss)
        test_losses.append(avg_val_loss)
        train_iou.append(avg_train_iou)
        val_iou.append(avg_val_iou)
        train_acc.append(avg_train_acc)
        val_acc.append(avg_val_acc)
        
        # 檢查是否有改善（基於 loss）
        if avg_val_loss < min_loss:
            print(f'\nLoss Decreasing.. {min_loss:.4f} >> {avg_val_loss:.4f}')
            min_loss = avg_val_loss
            not_improve = 0
            
            # 保存最佳 loss 模型
            print('Saving best loss model...')
            torch.save(model, 'TransUNet_best_loss.pt')
        else:
            not_improve += 1
            print(f'\nLoss Not Decrease for {not_improve} consecutive time(s)')
            if not_improve == 7:
                print('Loss not decrease for 7 consecutive times, Stop Training')
                break
            
        
        # 檢查是否有最佳 mIoU（另外保存）
        if avg_val_iou > best_iou:
            print(f'Best mIoU Improved: {best_iou:.4f} >> {avg_val_iou:.4f}')
            best_iou = avg_val_iou
            print('Saving best mIoU model...')
            torch.save(model, 'TransUNet_best_miou.pt')

        # 打印訓練信息
        epoch_time = (time.time() - since) / 60
        current_lr = get_lr(optimizer)
        
        print(f"\nEpoch: {e+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} (CE: {avg_train_loss_ce:.4f}, Dice: {avg_train_loss_dice:.4f})")
        print(f"  Val Loss:   {avg_val_loss:.4f} (CE: {avg_val_loss_ce:.4f}, Dice: {avg_val_loss_dice:.4f})")
        print(f"  Train mIoU: {avg_train_iou:.4f} | Val mIoU: {avg_val_iou:.4f}")
        print(f"  Train Acc:  {avg_train_acc:.4f} | Val Acc:  {avg_val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.2f}m")
        print("-" * 70)

    history = {
        'train_loss': train_losses,
        'val_loss': test_losses,
        'train_miou': train_iou,
        'val_miou': val_iou,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'lrs': lrs
    }
    
    total_time = (time.time() - fit_time) / 60
    print(f'\nTotal training time: {total_time:.2f} minutes')
    return history
