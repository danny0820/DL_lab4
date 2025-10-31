"""
TransUNet 模型實現
適配 DL_lab4 的訓練框架
"""

import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from collections import OrderedDict
import torch.nn.functional as F


# ============= ResNet Backbone =============

def np2th(weights, conv=False):
    """將 numpy 權重轉換為 torch 權重"""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    """標準化卷積層"""
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block"""
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


class ResNetV2(nn.Module):
    """Pre-activation ResNet V2 作為編碼器"""
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


# ============= Transformer Components =============

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, hidden_size, num_heads, attention_dropout_rate):
        super(Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    """MLP as used in Vision Transformer"""
    def __init__(self, hidden_size, mlp_dim, dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout_rate, attention_dropout_rate):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim, dropout_rate)
        self.attn = Attention(hidden_size, num_heads, attention_dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, hidden_size, num_layers, num_heads, mlp_dim, dropout_rate, attention_dropout_rate):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, num_heads, mlp_dim, dropout_rate, attention_dropout_rate)
            self.layer.append(layer)

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Embeddings(nn.Module):
    """將圖像轉換為 patch embeddings"""
    def __init__(self, img_size, hidden_size, dropout_rate, use_hybrid=True, resnet_num_layers=(3, 4, 9), resnet_width_factor=1):
        super(Embeddings, self).__init__()
        self.hybrid = use_hybrid
        img_size = _pair(img_size)

        if self.hybrid:
            # 使用 ResNet 作為 hybrid backbone
            # ResNet 會將圖像下採樣 16 倍，例如 512 -> 32
            # 然後使用 1x1 conv 進行 patch embedding
            grid_size = (16, 16)  # patch grid size after ResNet
            patch_size = (1, 1)  # 使用 1x1 conv 因為 ResNet 已經做了下採樣
            
            # ResNet 下採樣後的特徵圖大小
            feature_size = img_size[0] // 16  # 例如 512 // 16 = 32
            n_patches = feature_size * feature_size  # 32 * 32 = 1024
            
            self.hybrid_model = ResNetV2(block_units=resnet_num_layers, width_factor=resnet_width_factor)
            in_channels = self.hybrid_model.width * 16  # ResNet 輸出通道數
            
            self.patch_embeddings = Conv2d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_size
            )
        else:
            # 純 ViT，不使用 hybrid
            patch_size = _pair(16)
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(
                in_channels=3,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_size
            )

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
            
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        # 動態調整 position embeddings 以匹配實際 patch 數量
        B, n_patches, hidden = x.size()
        if n_patches != self.position_embeddings.size(1):
            # 使用插值調整 position embeddings 的大小
            pos_embed = self.position_embeddings.permute(0, 2, 1)  # (1, hidden, n_patches_orig)
            h_orig = w_orig = int(np.sqrt(self.position_embeddings.size(1)))
            h_new = w_new = int(np.sqrt(n_patches))
            pos_embed = pos_embed.reshape(1, hidden, h_orig, w_orig)
            pos_embed = F.interpolate(pos_embed, size=(h_new, w_new), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.reshape(1, hidden, h_new * w_new).permute(0, 2, 1)
            embeddings = x + pos_embed
        else:
            embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Transformer(nn.Module):
    """Vision Transformer"""
    def __init__(self, img_size, hidden_size, num_layers, num_heads, mlp_dim, 
                 dropout_rate, attention_dropout_rate, use_hybrid=True):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size, hidden_size, dropout_rate, use_hybrid)
        self.encoder = Encoder(hidden_size, num_layers, num_heads, mlp_dim, dropout_rate, attention_dropout_rate)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features


# ============= Decoder Components =============

class Conv2dReLU(nn.Sequential):
    """卷積 + 批次正規化 + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    """解碼器區塊"""
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # 確保尺寸匹配
            if x.size()[2:] != skip.size()[2:]:
                # 使用插值調整 skip 的大小以匹配 x
                skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    """TransUNet 解碼器"""
    def __init__(self, hidden_size, decoder_channels, skip_channels, n_skip):
        super().__init__()
        self.n_skip = n_skip
        
        head_channels = 512
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        # 根據 n_skip 重新選擇 skip channels
        if self.n_skip != 0:
            skip_channels_list = list(skip_channels)
            for i in range(4 - self.n_skip):
                skip_channels_list[3-i] = 0
        else:
            skip_channels_list = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) 
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels_list)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


# ============= Main Model =============

class TransUNet(nn.Module):
    """
    TransUNet 模型
    適配 DL_lab4 的訓練框架
    
    Args:
        img_ch: 輸入圖像通道數 (預設: 3)
        output_ch: 輸出類別數 (預設: 3)
        img_size: 輸入圖像大小 (預設: 512)
        hidden_size: Transformer hidden size (預設: 768)
        num_layers: Transformer layers 數量 (預設: 12)
        num_heads: Multi-head attention heads 數量 (預設: 12)
        mlp_dim: MLP dimension (預設: 3072)
        decoder_channels: 解碼器通道數 (預設: (256, 128, 64, 16))
        skip_channels: Skip connection 通道數 (預設: [512, 256, 64, 16])
        n_skip: 使用的 skip connection 數量 (預設: 3)
        use_hybrid: 是否使用 ResNet hybrid backbone (預設: True)
    """
    def __init__(
        self, 
        img_ch=3, 
        output_ch=3,
        img_size=512,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        dropout_rate=0.1,
        attention_dropout_rate=0.0,
        decoder_channels=(256, 128, 64, 16),
        skip_channels=[512, 256, 64, 16],
        n_skip=3,
        use_hybrid=True
    ):
        super(TransUNet, self).__init__()
        self.num_classes = output_ch
        self.img_size = img_size
        self.use_hybrid = use_hybrid
        
        # Transformer Encoder
        self.transformer = Transformer(
            img_size=img_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            use_hybrid=use_hybrid
        )
        
        # Decoder
        self.decoder = DecoderCup(
            hidden_size=hidden_size,
            decoder_channels=decoder_channels,
            skip_channels=skip_channels,
            n_skip=n_skip
        )
        
        # Segmentation Head
        self.segmentation_head = nn.Conv2d(
            decoder_channels[-1], 
            output_ch, 
            kernel_size=3, 
            padding=1
        )

    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: 輸入圖像 (B, C, H, W)
            
        Returns:
            logits: 分割結果 (B, num_classes, H, W)
        """
        # 如果是灰度圖，複製到3通道
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # Transformer Encoder
        x, features = self.transformer(x)  # (B, n_patch, hidden)
        
        # Decoder
        x = self.decoder(x, features)
        
        # Segmentation Head
        logits = self.segmentation_head(x)
        
        return logits


# ============= 配置函數 =============

def get_transunet_config(variant='R50-B16', num_classes=3, img_size=512):
    """
    獲取 TransUNet 配置
    
    Args:
        variant: 模型變體 ('R50-B16', 'ViT-B16', 'ViT-B32')
        num_classes: 類別數量
        img_size: 輸入圖像大小
        
    Returns:
        config: 模型配置字典
    """
    configs = {
        'R50-B16': {
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'mlp_dim': 3072,
            'decoder_channels': (256, 128, 64, 16),
            'skip_channels': [512, 256, 64, 16],
            'n_skip': 3,
            'use_hybrid': True,
        },
        'ViT-B16': {
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'mlp_dim': 3072,
            'decoder_channels': (256, 128, 64, 16),
            'skip_channels': [0, 0, 0, 0],
            'n_skip': 0,
            'use_hybrid': False,
        },
        'ViT-B32': {
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'mlp_dim': 3072,
            'decoder_channels': (256, 128, 64, 16),
            'skip_channels': [0, 0, 0, 0],
            'n_skip': 0,
            'use_hybrid': False,
        }
    }
    
    config = configs.get(variant, configs['R50-B16'])
    config['output_ch'] = num_classes
    config['img_size'] = img_size
    
    return config
