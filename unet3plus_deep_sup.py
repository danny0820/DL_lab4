"""
UNet 3+ with Deep Supervision
包含深度監督機制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷積區塊，包含多個卷積層、批次正規化和ReLU激活"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 is_bn=True, is_relu=True, n=2):
        super(ConvBlock, self).__init__()
        
        layers = []
        for i in range(1, n + 1):
            conv = nn.Conv2d(
                in_channels=in_channels if i == 1 else out_channels, 
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not is_bn
            )
            layers.append(conv)
            
            if is_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            
            if is_relu:
                layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)


class UNet3PlusDeepSup(nn.Module):
    """
    UNet 3+ with Deep Supervision
    具有全尺度跳躍連接和深度監督的 U-Net 變體
    """
    
    def __init__(self, img_ch=3, output_ch=3):
        super(UNet3PlusDeepSup, self).__init__()
        
        # 濾波器通道數設定
        self.filters = [64, 128, 256, 512, 1024]
        self.cat_channels = self.filters[0]  # 每個分支輸出 64 通道
        self.cat_blocks = len(self.filters)  # 5 個分支
        self.upsample_channels = self.cat_blocks * self.cat_channels  # 5 * 64 = 320

        # ==================== Encoder ====================
        # 下採樣路徑
        self.e1 = ConvBlock(img_ch, self.filters[0], n=2)
        
        self.e2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[0], self.filters[1], n=2)
        )
        
        self.e3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[1], self.filters[2], n=2)
        )
        
        self.e4 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[2], self.filters[3], n=2)
        )
        
        self.e5 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[3], self.filters[4], n=2)
        )

        # ==================== Decoder 4 ====================
        # 第4層解碼器：融合所有編碼器層的特徵
        self.d4_e1_to_d4 = ConvBlock(self.filters[0], self.cat_channels, n=1)  # 64->64
        self.d4_e2_to_d4 = ConvBlock(self.filters[1], self.cat_channels, n=1)  # 128->64
        self.d4_e3_to_d4 = ConvBlock(self.filters[2], self.cat_channels, n=1)  # 256->64
        self.d4_e4_to_d4 = ConvBlock(self.filters[3], self.cat_channels, n=1)  # 512->64
        self.d4_e5_to_d4 = ConvBlock(self.filters[4], self.cat_channels, n=1)  # 1024->64
        self.d4_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)  # 320->320

        # ==================== Decoder 3 ====================
        self.d3_e1_to_d3 = ConvBlock(self.filters[0], self.cat_channels, n=1)  # 64->64
        self.d3_e2_to_d3 = ConvBlock(self.filters[1], self.cat_channels, n=1)  # 128->64
        self.d3_e3_to_d3 = ConvBlock(self.filters[2], self.cat_channels, n=1)  # 256->64
        self.d3_d4_to_d3 = ConvBlock(self.upsample_channels, self.cat_channels, n=1)  # 320->64
        self.d3_e5_to_d3 = ConvBlock(self.filters[4], self.cat_channels, n=1)  # 1024->64
        self.d3_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)  # 320->320

        # ==================== Decoder 2 ====================
        self.d2_e1_to_d2 = ConvBlock(self.filters[0], self.cat_channels, n=1)  # 64->64
        self.d2_e2_to_d2 = ConvBlock(self.filters[1], self.cat_channels, n=1)  # 128->64
        self.d2_d3_to_d2 = ConvBlock(self.upsample_channels, self.cat_channels, n=1)  # 320->64
        self.d2_d4_to_d2 = ConvBlock(self.upsample_channels, self.cat_channels, n=1)  # 320->64
        self.d2_e5_to_d2 = ConvBlock(self.filters[4], self.cat_channels, n=1)  # 1024->64
        self.d2_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)  # 320->320

        # ==================== Decoder 1 ====================
        self.d1_e1_to_d1 = ConvBlock(self.filters[0], self.cat_channels, n=1)  # 64->64
        self.d1_d2_to_d1 = ConvBlock(self.upsample_channels, self.cat_channels, n=1)  # 320->64
        self.d1_d3_to_d1 = ConvBlock(self.upsample_channels, self.cat_channels, n=1)  # 320->64
        self.d1_d4_to_d1 = ConvBlock(self.upsample_channels, self.cat_channels, n=1)  # 320->64
        self.d1_e5_to_d1 = ConvBlock(self.filters[4], self.cat_channels, n=1)  # 1024->64
        self.d1_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)  # 320->320

        # ==================== 深度監督輸出層 ====================
        # 每個解碼器層都有一個獨立的輸出
        self.out_d2 = ConvBlock(self.upsample_channels, output_ch, n=1, is_bn=False, is_relu=False)
        self.out_d3 = ConvBlock(self.upsample_channels, output_ch, n=1, is_bn=False, is_relu=False)
        self.out_d4 = ConvBlock(self.upsample_channels, output_ch, n=1, is_bn=False, is_relu=False)
        self.out_e5 = ConvBlock(self.filters[4], output_ch, n=1, is_bn=False, is_relu=False)

        # ==================== 最終輸出層 ====================
        self.final = nn.Conv2d(self.upsample_channels, output_ch, kernel_size=1)

    def forward(self, x):
        """
        前向傳播函數
        e : 編碼器層
        d : 解碼器層
        
        訓練時返回 5 個輸出 (d1, d2, d3, d4, e5)
        推理時只返回主輸出 (d1)
        """
        # ==================== Encoder ====================
        e1 = self.e1(x)        # 64, H, W
        e2 = self.e2(e1)       # 128, H/2, W/2
        e3 = self.e3(e2)       # 256, H/4, W/4
        e4 = self.e4(e3)       # 512, H/8, W/8
        e5 = self.e5(e4)       # 1024, H/16, W/16

        # ==================== Decoder 4 ====================
        # 將所有編碼器特徵調整到 H/8 大小
        d4_e1 = F.max_pool2d(e1, 8)  # 下採樣到 H/8
        d4_e1 = self.d4_e1_to_d4(d4_e1)
        
        d4_e2 = F.max_pool2d(e2, 4)  # 下採樣到 H/8
        d4_e2 = self.d4_e2_to_d4(d4_e2)
        
        d4_e3 = F.max_pool2d(e3, 2)  # 下採樣到 H/8
        d4_e3 = self.d4_e3_to_d4(d4_e3)
        
        d4_e4 = self.d4_e4_to_d4(e4)  # 已經是 H/8
        
        d4_e5 = F.interpolate(e5, scale_factor=2, mode='bilinear', align_corners=True)  # 上採樣到 H/8
        d4_e5 = self.d4_e5_to_d4(d4_e5)
        
        # 拼接並卷積
        d4 = torch.cat([d4_e1, d4_e2, d4_e3, d4_e4, d4_e5], dim=1)
        d4 = self.d4_conv(d4)

        # ==================== Decoder 3 ====================
        # 將所有特徵調整到 H/4 大小
        d3_e1 = F.max_pool2d(e1, 4)
        d3_e1 = self.d3_e1_to_d3(d3_e1)
        
        d3_e2 = F.max_pool2d(e2, 2)
        d3_e2 = self.d3_e2_to_d3(d3_e2)
        
        d3_e3 = self.d3_e3_to_d3(e3)
        
        d3_d4 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        d3_d4 = self.d3_d4_to_d3(d3_d4)
        
        d3_e5 = F.interpolate(e5, scale_factor=4, mode='bilinear', align_corners=True)
        d3_e5 = self.d3_e5_to_d3(d3_e5)
        
        d3 = torch.cat([d3_e1, d3_e2, d3_e3, d3_d4, d3_e5], dim=1)
        d3 = self.d3_conv(d3)

        # ==================== Decoder 2 ====================
        # 將所有特徵調整到 H/2 大小
        d2_e1 = F.max_pool2d(e1, 2)
        d2_e1 = self.d2_e1_to_d2(d2_e1)
        
        d2_e2 = self.d2_e2_to_d2(e2)
        
        d2_d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2_d3 = self.d2_d3_to_d2(d2_d3)
        
        d2_d4 = F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=True)
        d2_d4 = self.d2_d4_to_d2(d2_d4)
        
        d2_e5 = F.interpolate(e5, scale_factor=8, mode='bilinear', align_corners=True)
        d2_e5 = self.d2_e5_to_d2(d2_e5)
        
        d2 = torch.cat([d2_e1, d2_e2, d2_d3, d2_d4, d2_e5], dim=1)
        d2 = self.d2_conv(d2)

        # ==================== Decoder 1 ====================
        # 將所有特徵調整到 H 大小
        d1_e1 = self.d1_e1_to_d1(e1)
        
        d1_d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1_d2 = self.d1_d2_to_d1(d1_d2)
        
        d1_d3 = F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True)
        d1_d3 = self.d1_d3_to_d1(d1_d3)
        
        d1_d4 = F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=True)
        d1_d4 = self.d1_d4_to_d1(d1_d4)
        
        d1_e5 = F.interpolate(e5, scale_factor=16, mode='bilinear', align_corners=True)
        d1_e5 = self.d1_e5_to_d1(d1_e5)
        
        d1 = torch.cat([d1_e1, d1_d2, d1_d3, d1_d4, d1_e5], dim=1)
        d1 = self.d1_conv(d1)

        # ==================== 輸出 ====================
        # 主輸出
        out = self.final(d1)
        
        # 深度監督輸出（僅在訓練時使用）
        if self.training:
            # 將所有輔助輸出上採樣到原始大小
            out_d2 = F.interpolate(self.out_d2(d2), scale_factor=2, mode='bilinear', align_corners=True)
            out_d3 = F.interpolate(self.out_d3(d3), scale_factor=4, mode='bilinear', align_corners=True)
            out_d4 = F.interpolate(self.out_d4(d4), scale_factor=8, mode='bilinear', align_corners=True)
            out_e5 = F.interpolate(self.out_e5(e5), scale_factor=16, mode='bilinear', align_corners=True)
            
            # 返回所有輸出：主輸出 + 4 個輔助輸出
            return out, out_d2, out_d3, out_d4, out_e5
        else:
            # 推理時只返回主輸出
            return out
