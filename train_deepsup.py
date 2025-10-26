"""
訓練模型的函數 - Deep Supervision 版本
支援 UNet3Plus with Deep Supervision 的多輸出損失計算
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


def fit_deepsup(epochs, model, train_loader, val_loader, criterion1, criterion2, optimizer, scheduler, device, 
                model_name='UNet3PlusDeepSup', patch=False):
    """
    訓練模型的主函數 - Deep Supervision 版本
    
    Args:
        epochs: 訓練輪數
        model: 模型（支援 Deep Supervision）
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器
        criterion1: 損失函數1 (Focal Loss 或 CrossEntropy)
        criterion2: 損失函數2 (Dice Loss)
        optimizer: 優化器
        scheduler: 學習率調度器
        device: 訓練設備
        model_name: 模型名稱，用於保存檔案
        patch: 是否使用patch訓練
    
    Returns:
        history: 包含訓練歷史的字典
    """
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    best_iou = 0.0  # 追蹤最佳 mIoU
    not_improve = 0  # 連續未改善的次數

    model.to(device)
    fit_time = time.time()
    
    # Deep Supervision 權重退火設置
    # 初始權重與最小權重
    init_weights = [1.0, 0.5, 0.25, 0.125, 0.0625]  # 對應 [d1, d2, d3, d4, e5]
    min_weights = [1.0, 0.0, 0.0, 0.0, 0.0]         # 最終只保留主分支
    print("=" * 70)
    print(f"Deep Supervision 訓練配置:")
    print(f"  - 模型: {model_name}")
    print(f"  - 輸出數量: {len(init_weights)} (主輸出 + {len(init_weights)-1} 個輔助輸出)")
    print(f"  - 初始損失權重: {init_weights}")
    print(f"  - 最小損失權重: {min_weights}")
    print("=" * 70)
    

    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0

        # 線性退火計算 ds_weights
        alpha = max(0, 1 - e / (epochs - 1))
        ds_weights = [min_w + (init_w - min_w) * alpha for init_w, min_w in zip(init_weights, min_weights)]
        print(f"[Epoch {e+1}/{epochs}] DS權重: {ds_weights}")

        # ==================== 訓練階段 ====================
        model.train()
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {e+1}/{epochs} [Train]")):
            image, mask = data

            # 將數據移到指定設備
            image = image.to(device)
            mask = mask.to(device)

            # 前向傳播 - Deep Supervision 返回多個輸出
            outputs = model(image)

            # 檢查是否為 Deep Supervision 模型（訓練時返回多個輸出）
            if isinstance(outputs, tuple) or isinstance(outputs, list):
                # Deep Supervision: 計算加權損失
                total_loss = 0
                for idx, output in enumerate(outputs):
                    output_loss = criterion1(output, mask) + criterion2(output, mask)
                    total_loss += ds_weights[idx] * output_loss
                loss = total_loss
                # 使用主輸出（第一個輸出）計算指標
                main_output = outputs[0]
                iou_score += mIoU(main_output, mask)
                accuracy += pixel_accuracy(main_output, mask)
            else:
                # 標準單一輸出（fallback）
                loss = criterion1(outputs, mask) + criterion2(outputs, mask)
                iou_score += mIoU(outputs, mask)
                accuracy += pixel_accuracy(outputs, mask)

            # 反向傳播
            loss.backward()
            # 更新權重
            optimizer.step()
            # 重置梯度
            optimizer.zero_grad()
            # 步進學習率
            lrs.append(get_lr(optimizer))
            scheduler.step()
            running_loss += loss.item()

        # ==================== 驗證階段 ====================
        model.eval()
        test_loss = 0
        test_accuracy = 0
        val_iou_score = 0
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader, desc=f"Epoch {e+1}/{epochs} [Val]")):
                image, mask = data

                # 將數據移到指定設備
                image = image.to(device)
                mask = mask.to(device)
                
                # 前向傳播 (驗證時 Deep Supervision 模型只返回主輸出)
                output = model(image)
                
                # 計算指標
                val_iou_score += mIoU(output, mask)
                test_accuracy += pixel_accuracy(output, mask)
                
                # 計算損失（驗證時只用主輸出）
                loss = criterion1(output, mask) + criterion2(output, mask)
                test_loss += loss.item()

        # 計算每個batch的平均值
        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(val_loader))

        # 計算當前 epoch 的平均驗證損失和 mIoU
        current_val_loss = test_loss/len(val_loader)
        current_val_iou = val_iou_score/len(val_loader)
        
        # 檢查是否有改善（基於 loss）
        if current_val_loss < min_loss:
            print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, current_val_loss))
            min_loss = current_val_loss
            not_improve = 0  # 重置連續未改善計數器
            
            # 保存最佳 loss 模型
            best_loss_path = f'{model_name}_best_loss.pt'
            print(f'Saving best loss model... -> {best_loss_path}')
            torch.save(model, best_loss_path)
        else:
            not_improve += 1  # 增加連續未改善次數
            print(f'Loss Not Decrease for {not_improve} consecutive time(s)')
            if not_improve >= 10:
                print("Early stopping triggered due to no improvement in loss for 10 consecutive epochs.")
                break  # 提前停止訓練
        
        # 檢查是否有最佳 mIoU（另外保存）
        if current_val_iou > best_iou:
            print('Best mIoU Improved: {:.3f} >> {:.3f}'.format(best_iou, current_val_iou))
            best_iou = current_val_iou
            
            # 保存最佳 mIoU 模型
            best_miou_path = f'{model_name}_best_miou.pt'
            print(f'Saving best mIoU model... -> {best_miou_path}')
            torch.save(model, best_miou_path)

        # 記錄指標
        val_iou.append(val_iou_score/len(val_loader))
        train_iou.append(iou_score/len(train_loader))
        train_acc.append(accuracy/len(train_loader))
        val_acc.append(test_accuracy/len(val_loader))
        
        print("Epoch:{}/{}..".format(e+1, epochs),
              "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
              "Val Loss: {:.3f}..".format(test_loss/len(val_loader)),
              "Train mIoU:{:.3f}..".format(iou_score/len(train_loader)),
              "Val mIoU: {:.3f}..".format(val_iou_score/len(val_loader)),
              "Train Acc:{:.3f}..".format(accuracy/len(train_loader)),
              "Val Acc:{:.3f}..".format(test_accuracy/len(val_loader)),
              "Time: {:.2f}m".format((time.time()-since)/60))

    history = {
        'train_loss': train_losses,
        'val_loss': test_losses,
        'train_miou': train_iou,
        'val_miou': val_iou,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'lrs': lrs
    }
    
    print('Total time: {:.2f} m'.format((time.time() - fit_time)/60))
    return history
