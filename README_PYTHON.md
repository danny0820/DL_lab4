# BCSS 乳腺癌語義分割專案

這個專案從 Jupyter Notebook 拆分而來，用於訓練 Attention U-Net 模型進行乳腺癌語義分割。

## 檔案結構

```
DL_lab4/
├── dataset.py      # 數據集類別和數據處理函數
├── model.py        # Attention U-Net 模型架構
├── metrics.py      # 損失函數和評估指標
├── train.py        # 訓練循環函數
├── utils.py        # 預測和可視化工具函數
├── main.py         # 主訓練執行檔案
├── predict.py      # 測試集預測執行檔案
└── BCSS/          # 數據集資料夾
    ├── train/
    ├── train_mask/
    ├── val/
    ├── val_mask/
    └── test/
```

## 各檔案說明

### dataset.py
- `create_df()`: 創建包含圖像ID的DataFrame
- `BCSSDataset`: 訓練和驗證數據集類別
- `BCSSTestDataset`: 測試數據集類別

### model.py
- `ConvBlock`: 卷積區塊
- `UpConv`: 上採樣卷積區塊
- `AttentionBlock`: 注意力機制區塊
- `AttentionUNet`: Attention U-Net 主模型

### metrics.py
- `pixel_accuracy()`: 計算像素準確率
- `mIoU()`: 計算平均交並比
- `DiceLoss`: Dice損失函數類別

### train.py
- `get_lr()`: 獲取當前學習率
- `fit()`: 主訓練循環函數

### utils.py
- `predict_image()`: 預測單張圖像
- `plot_loss()`: 繪製損失曲線
- `plot_score()`: 繪製mIoU曲線
- `plot_acc()`: 繪製準確率曲線

### main.py
主訓練執行檔案，整合所有模組並執行完整的訓練流程

### predict.py
測試集預測執行檔案，載入訓練好的模型並生成Kaggle提交檔案

## 使用方法

### 1. 訓練模型

```bash
python main.py
```

這將會：
- 載入訓練和驗證數據
- 初始化 Attention U-Net 模型
- 訓練 100 個 epoch
- 每5次損失下降時保存模型
- 訓練完成後保存最終模型為 `Unet-Resnet.pt`
- 顯示訓練過程的損失、mIoU 和準確率曲線

### 2. 預測測試集

```bash
python predict.py
```

這將會：
- 載入訓練好的模型 `Unet-Resnet.pt`
- 對測試集進行預測
- 生成 `output.csv` 供 Kaggle 提交

## 訓練參數

在 `main.py` 中可以調整以下超參數：

```python
max_lr = 1e-3           # 最大學習率
epoch = 100             # 訓練輪數
weight_decay = 1e-4     # 權重衰減
batch_size = 32         # 批次大小
```

## GPU 設置

程式碼會自動偵測可用的 GPU 並使用：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # 使用 GPU 1 和 2
```

如果偵測到多個 GPU，會自動使用 `DataParallel` 進行多GPU訓練。

## 數據增強

訓練集使用的數據增強包括：
- 隨機水平翻轉 (p=0.5)
- 隨機垂直翻轉 (p=0.5)
- 隨機旋轉 (±90度)
- 標準化 (ImageNet 均值和標準差)

驗證集和測試集僅使用標準化。

## 模型保存

模型會在以下情況保存：
1. 每5次驗證損失下降時保存為 `Unet-Mobilenet_v2_mIoU-{score}.pt`
2. 訓練完成後保存最終模型為 `Unet-Resnet.pt`

## 早停機制

如果驗證損失連續 7 個 epoch 沒有下降，訓練會自動停止。

## 依賴套件

確保已安裝以下套件：
- torch
- torchvision
- numpy
- pandas
- matplotlib
- opencv-python (cv2)
- Pillow
- tqdm
- albumentations

## 注意事項

1. 確保 BCSS 資料夾結構正確
2. 確保有足夠的 GPU 記憶體（建議至少 8GB）
3. 訓練過程會自動使用 tqdm 顯示進度條
4. 所有程式碼保持與原 Notebook 一致，未做任何邏輯修改

## Kaggle 提交

訓練完成後，執行 `predict.py` 生成 `output.csv`，然後：
1. 前往 Kaggle 競賽頁面
2. 點擊 "Submit Predictions"
3. 上傳 `output.csv`
4. 系統會自動計算準確率並更新排行榜
