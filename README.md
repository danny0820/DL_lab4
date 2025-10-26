# DL_lab4 - 乳腺癌語義分割專案

本資料夾為醫學影像分割（乳腺癌）任務的完整專案，包含 Attention U-Net、UNet3Plus 及 Deep Supervision 訓練、推論、後處理等腳本。

## 目錄結構

- `main.py`、`main_deepsup.py`：主訓練腳本（含 Deep Supervision 版本）
- `train.py`、`train_deepsup.py`：訓練流程與函數
- `predict.py`：模型推論與產生 output.csv
- `metrics.py`：分割評估指標（Dice、mIoU 等）
- `utils.py`：輔助函數（繪圖、資料處理等）
- `dataset.py`：自訂資料集類別
- `model.py`、`attention_unet_deep_sup.py`、`unet3plus.py` 等：模型架構
- `CRF_Postprocessing.ipynb`：CRF 後處理範例
- `Lab4.ipynb`：完整分割流程範例
- `BCSS/`：資料集資料夾（建議勿上傳）
- `*.pt`：模型權重檔案
- `.gitignore`：已排除大型資料與中間檔案

## 快速開始

### 1. 安裝依賴
建議使用 Python 3.10+，安裝必要套件：

```bash
pip install -r requirements.txt
# 或手動安裝
pip install torch torchvision albumentations pandas tqdm pillow
```

### 2. 訓練模型

```bash
python main.py           # Attention U-Net
python main_deepsup.py   # Attention U-Net Deep Supervision
```

### 3. 推論產生 output.csv

```bash
python predict.py
```

可於 predict.py 內修改 `CUDA_VISIBLE_DEVICES` 指定 GPU。

### 4. CRF 後處理
請參考 `CRF_Postprocessing.ipynb` 或將 CRF 流程整合至 predict.py。



