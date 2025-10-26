# UNet3Plus Deep Supervision 訓練指南

## 📁 文件說明

本項目新增了專門用於 **Deep Supervision** 訓練的文件：

### 核心文件

- **`main_deepsup.py`**: Deep Supervision 版本的主訓練腳本
- **`train_deepsup.py`**: Deep Supervision 版本的訓練函數
- **`unet3plus_deep_sup.py`**: UNet3Plus with Deep Supervision 模型定義

### 原始文件（保持不變）

- **`main.py`**: 標準訓練腳本（用於 AttentionUNet 等模型）
- **`train.py`**: 標準訓練函數
- **`unet3plus.py`**: 標準 UNet3Plus 模型

---

## 🚀 快速開始

### 使用 Deep Supervision 訓練

```bash
python main_deepsup.py
```

### 使用標準模型訓練

```bash
python main.py
```

---

## 🔍 Deep Supervision 原理

### 什麼是 Deep Supervision？

Deep Supervision（深度監督）是一種訓練技術，在模型的多個中間層都添加輸出分支，並對每個分支計算損失。

### 優勢

1. **改善梯度流動**: 為深層網路提供更好的梯度傳播
2. **多尺度監督**: 強制中間層學習有意義的特徵表示
3. **防止過擬合**: 正則化效果，提升泛化能力
4. **性能提升**: 通常能獲得更好的分割精度

### 模型輸出

- **訓練模式**: 返回 5 個輸出
  - `out_d1`: 主輸出 (H × W)
  - `out_d2`: 解碼器第2層輸出 (H/2 × W/2 → 上採樣到 H × W)
  - `out_d3`: 解碼器第3層輸出 (H/4 × W/4 → 上採樣到 H × W)
  - `out_d4`: 解碼器第4層輸出 (H/8 × W/8 → 上採樣到 H × W)
  - `out_e5`: 編碼器第5層輸出 (H/16 × W/16 → 上採樣到 H × W)

- **推理模式**: 只返回主輸出 `out_d1`

---

## ⚙️ 損失計算

### 加權策略

Deep Supervision 使用遞減的權重對多個輸出進行加權：

```python
ds_weights = [1.0, 0.8, 0.6, 0.4, 0.2]
```

- **主輸出 (d1)**: 權重 1.0 - 最重要
- **輔助輸出 (d2)**: 權重 0.8
- **輔助輸出 (d3)**: 權重 0.6
- **輔助輸出 (d4)**: 權重 0.4
- **最深層 (e5)**: 權重 0.2

### 總損失公式

```
Total Loss = Σ (weight_i × (FocalLoss_i + DiceLoss_i))
           = 1.0 × Loss(d1) + 0.8 × Loss(d2) + 0.6 × Loss(d3) + 0.4 × Loss(d4) + 0.2 × Loss(e5)
```

---

## 📊 訓練配置

### 超參數設置

```python
max_lr = 0.001
epoch = 100
batch_size = 32
weight_decay = 0.0001
```

### 損失函數

- **Focal Loss**: `alpha=1.0, gamma=2.0` - 處理類別不平衡
- **Dice Loss**: 提升分割邊界精度

### 優化器

- **AdamW**: 帶權重衰減的 Adam 優化器
- **OneCycleLR**: 循環學習率調度器

---

## 💾 模型保存

訓練過程會自動保存三個模型：

1. **`UNet3PlusDeepSup_best_loss.pt`**: 驗證損失最低的模型
2. **`UNet3PlusDeepSup_best_miou.pt`**: mIoU 最高的模型
3. **`UNet3PlusDeepSup_final.pt`**: 最終訓練完成的模型

---

## 🔧 自定義配置

### 修改 GPU 設置

在 `main_deepsup.py` 中修改：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # 使用 GPU 1 和 2
```

### 修改損失權重

在 `train_deepsup.py` 中修改：

```python
ds_weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # 自定義權重
```

### 修改 Batch Size

在 `main_deepsup.py` 中修改：

```python
batch_size = 32  # 根據 GPU 記憶體調整
```

---

## 📈 訓練輸出

訓練過程會顯示詳細信息：

```
======================================================================
Deep Supervision 訓練配置:
  - 模型: UNet3PlusDeepSup
  - 輸出數量: 5 (主輸出 + 4 個輔助輸出)
  - 損失權重: [1.0, 0.8, 0.6, 0.4, 0.2]
  - 總權重: 3.0
======================================================================

Epoch:1/100.. Train Loss: 0.845.. Val Loss: 0.723.. Train mIoU:0.456.. Val mIoU: 0.512.. ...
```

---

## 🧪 測試 Deep Supervision 邏輯

運行測試腳本驗證模型：

```python
import torch
from unet3plus_deep_sup import UNet3PlusDeepSup

model = UNet3PlusDeepSup(img_ch=3, output_ch=3)

# 訓練模式 - 返回多個輸出
model.train()
x = torch.randn(2, 3, 256, 256)
outputs = model(x)
print(f"訓練模式輸出數量: {len(outputs)}")  # 5

# 推理模式 - 返回單一輸出
model.eval()
output = model(x)
print(f"推理模式輸出形狀: {output.shape}")  # torch.Size([2, 3, 256, 256])
```

---

## 🔗 相關文件

- `UNET3PLUS_README.md`: UNet3Plus 模型的詳細說明
- `README_PYTHON.md`: Python 環境設置說明

---

## 📝 參考文獻

UNet 3+ 論文:
> UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation
> IEEE Access, 2020

Deep Supervision:
> Deeply-Supervised Nets
> AISTATS, 2015

---

## ✅ MCP 驗證結果

所有文件已通過以下檢查：

- ✓ 語法檢查 (Syntax Check)
- ✓ 類型檢查 (Type Check)
- ✓ 邏輯驗證 (Logic Verification)
- ✓ 實際運行測試 (Runtime Test)

---

**最後更新**: 2025-10-24
