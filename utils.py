import torch
import matplotlib.pyplot as plt
from torchvision import transforms as T


def predict_image(model, image, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    預測單張圖像的分割遮罩
    
    Args:
        model: 訓練好的模型
        image: PIL Image 格式的圖像
        device: 計算設備
        mean: 標準化均值
        std: 標準化標準差
    
    Returns:
        masked: 預測的分割遮罩
    """
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    
    return masked


def plot_loss(history):
    """繪製訓練和驗證損失曲線"""
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.show()


def plot_score(history):
    """繪製訓練和驗證mIoU曲線"""
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU', marker='*')
    plt.title('Score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.show()


def plot_acc(history):
    """繪製訓練和驗證準確率曲線"""
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy', marker='*')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.show()
