# UNet3Plus 模型說明

## 檔案結構

```
DL_lab4/
├── unet3plus_base.py          # UNet3Plus 基礎版本
├── unet3plus_deep_sup.py      # UNet3Plus 深度監督版本
├── test_unet3plus.py          # 模型測試腳本
└── unet3plus.py               # 原始合併版本（保留參考）
```

## 模型說明

### 1. UNet3Plus (基礎版本) - `unet3plus_base.py`

**特點：**
- 標準 UNet3+ 架構
- 全尺度跳躍連接（Full-scale skip connections）
- 5 層編碼器-解碼器結構
- 只有一個最終輸出

**參數量：** 26,963,331

**使用場景：**
- 標準的語義分割任務
- 不需要深度監督的場景
- 推理速度要求較高的場景

### 2. UNet3PlusDeepSup (深度監督版本) - `unet3plus_deep_sup.py`

**特點：**
- UNet3+ 架構 + 深度監督機制
- 全尺度跳躍連接
- 5 層編碼器-解碼器結構
- **訓練時：** 返回 5 個輸出（主輸出 + 4 個輔助輸出）
- **推理時：** 只返回主輸出

**參數量：** 27,016,911

**使用場景：**
- 需要更好收斂性能的訓練
- 醫學影像分割等精細任務
- 有足夠計算資源的場景

**深度監督的優勢：**
- 幫助梯度更好地回傳到淺層
- 加速收斂
- 提高分割精度
- 緩解梯度消失問題

## 使用方法

### 方法 1: 在 main.py 中直接替換模型

#### 使用基礎版本：

```python
# 在 main.py 中修改
from unet3plus_base import UNet3Plus

# 初始化模型
model = UNet3Plus(img_ch=3, output_ch=3)
```

#### 使用深度監督版本：

```python
# 在 main.py 中修改
from unet3plus_deep_sup import UNet3PlusDeepSup

# 初始化模型
model = UNet3PlusDeepSup(img_ch=3, output_ch=3)
```

### 方法 2: 修改 train.py 以支持深度監督

如果使用深度監督版本，需要修改訓練循環來處理多個輸出。

#### 修改 train.py 中的損失計算：

```python
# 原始版本（單輸出）
def fit(epoch, model, train_loader, val_loader, criterion1, criterion2, optimizer, scheduler, device):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss1 = criterion1(output, target)
        loss2 = criterion2(output, target)
        loss = loss1 + loss2
        # ... 後續代碼

# 深度監督版本（多輸出）
def fit(epoch, model, train_loader, val_loader, criterion1, criterion2, optimizer, scheduler, device):
    for batch_idx, (data, target) in enumerate(train_loader):
        outputs = model(data)
        
        if isinstance(outputs, tuple):  # 訓練模式，多個輸出
            # 主輸出
            main_output = outputs[0]
            # 輔助輸出
            aux_outputs = outputs[1:]
            
            # 計算主輸出損失
            loss1 = criterion1(main_output, target)
            loss2 = criterion2(main_output, target)
            main_loss = loss1 + loss2
            
            # 計算輔助輸出損失（權重較小）
            aux_loss = 0
            for aux_output in aux_outputs:
                aux_loss += criterion1(aux_output, target) + criterion2(aux_output, target)
            aux_loss = aux_loss / len(aux_outputs) * 0.4  # 輔助損失權重 0.4
            
            # 總損失
            loss = main_loss + aux_loss
        else:  # 推理模式，單個輸出
            loss1 = criterion1(outputs, target)
            loss2 = criterion2(outputs, target)
            loss = loss1 + loss2
        
        # ... 後續代碼
```

### 方法 3: 簡單替換（推薦）

如果不想修改訓練代碼，可以在訓練時設置模型為 eval 模式來只輸出主結果：

```python
from unet3plus_deep_sup import UNet3PlusDeepSup

model = UNet3PlusDeepSup(img_ch=3, output_ch=3)

# 訓練時強制只返回主輸出（不推薦，失去深度監督優勢）
model.eval()  # 這樣訓練時也只返回單個輸出
```

## 模型輸出說明

### UNet3Plus (基礎版本)

```python
model = UNet3Plus(3, 3)
x = torch.randn(2, 3, 256, 256)

output = model(x)
# 輸出形狀: torch.Size([2, 3, 256, 256])
```

### UNet3PlusDeepSup (深度監督版本)

```python
model = UNet3PlusDeepSup(3, 3)
x = torch.randn(2, 3, 256, 256)

# 訓練模式
model.train()
outputs = model(x)  # 返回元組，包含 5 個輸出
# outputs[0]: 主輸出 (d1)      - torch.Size([2, 3, 256, 256])
# outputs[1]: 輔助輸出 (d2)    - torch.Size([2, 3, 256, 256])
# outputs[2]: 輔助輸出 (d3)    - torch.Size([2, 3, 256, 256])
# outputs[3]: 輔助輸出 (d4)    - torch.Size([2, 3, 256, 256])
# outputs[4]: 輔助輸出 (e5)    - torch.Size([2, 3, 256, 256])

# 推理模式
model.eval()
output = model(x)  # 只返回主輸出
# 輸出形狀: torch.Size([2, 3, 256, 256])
```

## 測試模型

執行測試腳本來驗證模型是否正常工作：

```bash
cd /danny/DL_lab4
python test_unet3plus.py
```

## 快速開始

**最簡單的替換方法（不需要修改訓練代碼）：**

在 `main.py` 中找到這一行：

```python
from model import AttentionUNet
model = AttentionUNet(3, 3)
```

替換為：

```python
from unet3plus_base import UNet3Plus
model = UNet3Plus(3, 3)
```

就可以直接訓練了！

## 性能比較

| 模型 | 參數量 | 訓練輸出 | 推理輸出 | 訓練速度 | 收斂速度 |
|------|--------|----------|----------|----------|----------|
| AttentionUNet | ~31M | 1 | 1 | 快 | 中 |
| UNet3Plus | ~27M | 1 | 1 | 中 | 中 |
| UNet3PlusDeepSup | ~27M | 5 | 1 | 較慢 | 快 |

## 注意事項

1. **GPU 記憶體：** 深度監督版本在訓練時需要更多 GPU 記憶體（需要計算 5 個輸出的損失）
2. **訓練速度：** 深度監督版本訓練速度較慢，但收斂更快，總訓練時間可能更短
3. **推理速度：** 兩個版本推理速度相同（都只輸出主結果）
4. **精度：** 深度監督版本通常能達到更好的分割精度

## 參考資料

- 原始論文: UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation
- GitHub: https://github.com/ZJUGiveLab/UNet-Version
