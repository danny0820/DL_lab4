"""
Attention U-Net with Deep Supervision
包含注意力機制和深度監督的 U-Net 變體
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷積區塊，包含兩個卷積層、批次正規化和ReLU激活"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    """上採樣卷積區塊"""
    
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """注意力區塊，包含可學習參數"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNetDeepSup(nn.Module):
    """
    Attention U-Net with Deep Supervision
    結合注意力機制和深度監督的 U-Net 模型
    """
    
    def __init__(self, img_ch=3, output_ch=3):
        super(AttentionUNetDeepSup, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ==================== 編碼器路徑 ====================
        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        # ==================== 解碼器路徑 ====================
        # Decoder 5
        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        # Decoder 4
        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        # Decoder 3
        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        # Decoder 2
        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        # ==================== 深度監督輸出層 ====================
        # 每個解碼器層都有一個獨立的輸出分支
        self.out_d2 = nn.Conv2d(128, output_ch, kernel_size=1, stride=1, padding=0)
        self.out_d3 = nn.Conv2d(256, output_ch, kernel_size=1, stride=1, padding=0)
        self.out_d4 = nn.Conv2d(512, output_ch, kernel_size=1, stride=1, padding=0)
        self.out_e5 = nn.Conv2d(1024, output_ch, kernel_size=1, stride=1, padding=0)

        # ==================== 最終主輸出層 ====================
        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        前向傳播函數
        e : 編碼器層
        d : 解碼器層  
        s : 從編碼器到解碼器的跳躍連接（經過注意力機制加權）
        
        訓練時返回 5 個輸出 (主輸出, d2, d3, d4, e5)
        推理時只返回主輸出
        """
        # ==================== 編碼器路徑 ====================
        e1 = self.Conv1(x)       # 64, H, W

        e2 = self.MaxPool(e1)    # 64, H/2, W/2
        e2 = self.Conv2(e2)      # 128, H/2, W/2

        e3 = self.MaxPool(e2)    # 128, H/4, W/4
        e3 = self.Conv3(e3)      # 256, H/4, W/4

        e4 = self.MaxPool(e3)    # 256, H/8, W/8
        e4 = self.Conv4(e4)      # 512, H/8, W/8

        e5 = self.MaxPool(e4)    # 512, H/16, W/16
        e5 = self.Conv5(e5)      # 1024, H/16, W/16

        # ==================== 解碼器路徑 ====================
        # Decoder 5: H/16 -> H/8
        d5 = self.Up5(e5)                                    # 512, H/8, W/8
        s4 = self.Att5(gate=d5, skip_connection=e4)         # 注意力加權的跳躍連接
        d5 = torch.cat((s4, d5), dim=1)                     # 1024, H/8, W/8
        d5 = self.UpConv5(d5)                               # 512, H/8, W/8

        # Decoder 4: H/8 -> H/4
        d4 = self.Up4(d5)                                    # 256, H/4, W/4
        s3 = self.Att4(gate=d4, skip_connection=e3)         # 注意力加權的跳躍連接
        d4 = torch.cat((s3, d4), dim=1)                     # 512, H/4, W/4
        d4 = self.UpConv4(d4)                               # 256, H/4, W/4

        # Decoder 3: H/4 -> H/2
        d3 = self.Up3(d4)                                    # 128, H/2, W/2
        s2 = self.Att3(gate=d3, skip_connection=e2)         # 注意力加權的跳躍連接
        d3 = torch.cat((s2, d3), dim=1)                     # 256, H/2, W/2
        d3 = self.UpConv3(d3)                               # 128, H/2, W/2

        # Decoder 2: H/2 -> H
        d2 = self.Up2(d3)                                    # 64, H, W
        s1 = self.Att2(gate=d2, skip_connection=e1)         # 注意力加權的跳躍連接
        d2 = torch.cat((s1, d2), dim=1)                     # 128, H, W
        d2 = self.UpConv2(d2)                               # 64, H, W

        # ==================== 輸出 ====================
        # 主輸出
        out = self.Conv(d2)  # output_ch, H, W

        # 深度監督輸出（僅在訓練時使用）
        if self.training:
            # 在解碼器的不同階段提取輔助輸出
            # 注意：這些輸出是在 UpConv 之前的拼接特徵圖
            
            # d2 輔助輸出 (使用拼接後的特徵，d2 尺寸為 128 channels)
            out_d2 = self.out_d2(torch.cat((s1, self.Up2(d3)), dim=1))  # H, W
            
            # d3 輔助輸出 (使用拼接後的特徵，d3 尺寸為 256 channels)
            out_d3 = self.out_d3(torch.cat((s2, self.Up3(d4)), dim=1))  # H/2, W/2
            out_d3 = F.interpolate(out_d3, scale_factor=2, mode='bilinear', align_corners=True)  # 上採樣到 H, W
            
            # d4 輔助輸出 (使用拼接後的特徵，d4 尺寸為 512 channels)
            out_d4 = self.out_d4(torch.cat((s3, self.Up4(d5)), dim=1))  # H/4, W/4
            out_d4 = F.interpolate(out_d4, scale_factor=4, mode='bilinear', align_corners=True)  # 上採樣到 H, W
            
            # e5 輔助輸出 (瓶頸層)
            out_e5 = self.out_e5(e5)  # H/16, W/16
            out_e5 = F.interpolate(out_e5, scale_factor=16, mode='bilinear', align_corners=True)  # 上採樣到 H, W
            
            # 返回所有輸出：主輸出 + 4 個輔助輸出
            return out, out_d2, out_d3, out_d4, out_e5
        else:
            # 推理時只返回主輸出
            return out


# ==================== 測試模型 ====================
if __name__ == "__main__":
    # 創建模型實例
    model = AttentionUNetDeepSup(img_ch=3, output_ch=3)
    
    # 設定為訓練模式
    model.train()
    
    # 創建隨機輸入
    x = torch.randn(2, 3, 256, 256)  # batch_size=2, channels=3, height=256, width=256
    
    # 前向傳播
    outputs = model(x)
    
    print("=== 訓練模式輸出 ===")
    print(f"主輸出 (out): {outputs[0].shape}")
    print(f"輔助輸出 d2 (out_d2): {outputs[1].shape}")
    print(f"輔助輸出 d3 (out_d3): {outputs[2].shape}")
    print(f"輔助輸出 d4 (out_d4): {outputs[3].shape}")
    print(f"輔助輸出 e5 (out_e5): {outputs[4].shape}")
    
    # 設定為評估模式
    model.eval()
    
    # 前向傳播
    output = model(x)
    
    print("\n=== 評估模式輸出 ===")
    print(f"主輸出: {output.shape}")
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== 模型參數 ===")
    print(f"總參數量: {total_params:,}")
    print(f"可訓練參數量: {trainable_params:,}")
