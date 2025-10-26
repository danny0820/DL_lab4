import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_accuracy(output, mask):
    """計算像素準確率"""
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = (output == mask).float()
        accuracy = correct.sum() / correct.numel()
        accuracy = accuracy.item()
    return accuracy


def mIoU(pred_mask, mask, n_classes=3, ignore_class=0):
    """
    For batch input, calculates per-image mIoU and averages across the batch.
    """
    with torch.no_grad():
        probs = F.softmax(pred_mask, dim=1)
        preds = torch.argmax(probs, dim=1)  # (B, H, W)

        batch_size = preds.shape[0]
        batch_miou_scores = []

        # Calculate mIoU for each image in the batch (per-image averaging)
        for b in range(batch_size):
            pred_img = preds[b]  # (H, W)
            mask_img = mask[b]   # (H, W)

            iou_list = []
            
            for cls in range(n_classes):
                pred_c = (pred_img == cls)
                label_c = (mask_img == cls)

                intersection = (pred_c & label_c).sum().float()
                union = (pred_c | label_c).sum().float()

                # Skip if union is 0 (class doesn't exist in both GT and Pred)
                if union == 0:
                    continue

                iou = intersection / union

                if cls != ignore_class:
                    iou_list.append(iou)

            # Calculate mIoU for this image
            if len(iou_list) > 0:
                img_miou = torch.stack(iou_list).mean()
                batch_miou_scores.append(img_miou)

        # Return mean of per-image mIoU scores
        if len(batch_miou_scores) > 0:
            return float(torch.stack(batch_miou_scores).mean().item())
        else:
            return 0.0

class DiceLoss(nn.Module):
    """
    Dice loss
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, eps=1e-6):
        """
        Calculation of dice loss

        :param inputs: model predictions
        :param targets: target values
        :param eps: stability factor, defaults to 1e-6
        :return: loss value
        """
        # 對輸入應用 softmax 得到概率分佈
        inputs = F.softmax(inputs, dim=1)
        
        # 將目標標籤轉換為 one-hot 編碼
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 將張量展平以便計算
        inputs = inputs.contiguous().view(inputs.size(0), inputs.size(1), -1)
        targets_one_hot = targets_one_hot.contiguous().view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # 計算每個類別的交集和聯集
        intersection = (inputs * targets_one_hot).sum(dim=2)
        union = inputs.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        # 計算每個類別的 dice 係數
        dice = (2.0 * intersection + eps) / (union + eps)
        
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for semantic segmentation tasks.
    
    Loss(x, class) = - alpha * (1 - softmax(x)[class])^gamma * log(softmax(x)[class])
    
    Args:
        alpha: 類別權重係數，預設為 1.0
        gamma: 聚焦參數，預設為 2.0。gamma 越大，對簡單樣本的懲罰越小
        reduction: 'mean' 或 'sum'
    """
    def __init__(self, alpha, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型輸出 (B, C, H, W) - logits
            targets: 真實標籤 (B, H, W) - 類別索引
        """
        # 計算 log softmax
        log_probs = F.log_softmax(inputs, dim=1)  # (B, C, H, W)
        
        # 獲取每個像素對應類別的 log probability
        # targets: (B, H, W) -> (B, 1, H, W)
        targets_unsqueezed = targets.unsqueeze(1)
        
        # 使用 gather 提取對應類別的 log probability
        # log_pt: (B, 1, H, W)
        log_pt = log_probs.gather(1, targets_unsqueezed)
        
        # 移除多餘的維度: (B, 1, H, W) -> (B, H, W)
        log_pt = log_pt.squeeze(1)
        
        # 計算 pt (probability)
        pt = torch.exp(log_pt)
        
        # 計算 focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # 計算 focal loss
        focal_loss = -self.alpha * focal_weight * log_pt
        
        # 應用 reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

