# import torch
# import torch.nn as nn


# class ConvBlock(nn.Module):

#     def __init__(self, in_ch, out_ch, norm_layer, momentum=0.01):
#         super().__init__()
#         if norm_layer is nn.Identity:
#             bias = True
#         else:
#             bias = False

#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias),
#             norm_layer(out_ch, momentum=momentum),
#             nn.ReLU(),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=bias),
#             norm_layer(out_ch, momentum=momentum),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         return self.block(x)


# class DownsampleBlock(nn.Module):

#     def __init__(self, in_ch: int, out_ch: int, norm_layer=None):
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d

#         self.block = ConvBlock(in_ch, out_ch, norm_layer=norm_layer)
#         self.downsample = nn.MaxPool2d(kernel_size=2)

#     def forward(self, x):
#         x = self.block(x)
#         y = x
#         x = self.downsample(x)
#         return x, y


# class UpsampleBlock(nn.Module):

#     def __init__(self, in_ch: int, out_ch: int, norm_layer=None):
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d

#         self.block = ConvBlock(in_ch, out_ch, norm_layer=norm_layer)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

#     def forward(self, x, y):
#         x = self.upsample(x)
#         x = torch.cat([x, y], dim=1)
#         x = self.block(x)
#         return x


# class UNet(nn.Module):

#     def __init__(self, channels, n_layers: int, norm_layer=None):
#         super().__init__()

#         self.downblocks = nn.ModuleList()
#         self.upblocks = nn.ModuleList()
#         self.n_layers = n_layers

#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d

#         for i in range(n_layers):
#             block = DownsampleBlock(channels[i], channels[i + 1], norm_layer)
#             self.downblocks.append(block)

#         bottom_in = channels[n_layers]
#         bottom_out = channels[n_layers + 1]
#         self.bottom_block = ConvBlock(bottom_in, bottom_out, norm_layer=norm_layer)

#         for i in range(n_layers):
#             block = UpsampleBlock(channels[i + 1] + channels[i + 2], channels[i + 1], norm_layer)
#             self.upblocks.append(block)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         features = []
#         for i in range(self.n_layers):
#             x, y = self.downblocks[i](x)
#             features.append(y)
#         x = self.bottom_block(x)
#         for i in range(self.n_layers - 1, -1, -1):
#             x = self.upblocks[i](x, features[i])
#         return x
import torch
import torch.nn as nn

# ================= 基础模块 (保持原有风格) =================

# 基础卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, momentum=0.01):
        super().__init__()
        if norm_layer is nn.Identity:
            bias = True
        else:
            bias = False
        # padding=1保持空间尺寸不变
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias),
            norm_layer(out_ch, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=bias),
            norm_layer(out_ch, momentum=momentum),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

# 下采样模块
class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d 
        self.block = ConvBlock(in_ch, out_ch, norm_layer=norm_layer)
        # 2*2最大池化 改变（H，W）
        self.downsample = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # x 先经过卷积块提取特征，作为Skip Connection的 y
        # 然后 x 进行池化下采样
        x = self.block(x)
        y = x
        x = self.downsample(x)
        return x, y

# 上采样模块
class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # in_ch 是指拼接后的总通道数
        self.block = ConvBlock(in_ch, out_ch, norm_layer=norm_layer)
        # 上采样 空间尺寸翻倍 通道不变
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, y):
        # 先上采样
        x = self.upsample(x)
        # 拼接跳跃链接
        x = torch.cat([x, y], dim=1)
        # 卷积融合
        x = self.block(x)
        return x

# ================= 主模型结构 (不含光谱残差) =================

class DualHeadUNet(nn.Module):

    def __init__(self, norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # 1. 定义通道参数 (根据图片结构)
        # Encoder: 3 -> 32 -> 64 -> 128 -> 256 -> 512 -> [Bottleneck: 1024]
        self.enc_channels = [3, 32, 64, 128, 256, 512] 
        bottleneck_in = 512
        bottleneck_out = 1024
        
        # Decoder 1 (Depth): 1024 -> ... -> 16 -> Output(1)
        self.depth_channels = [512, 256, 128, 64, 16] 
        
        # Decoder 2 (HS Image): 1024 -> ... -> 32 -> Output(25)
        self.hs_channels = [512, 256, 128, 64, 32]

        self.n_layers = 5 # 对应5次下采样

        # --- 构建 Encoder ---
        self.downblocks = nn.ModuleList()
        for i in range(self.n_layers):
            block = DownsampleBlock(self.enc_channels[i], self.enc_channels[i + 1], norm_layer)
            self.downblocks.append(block)

        # --- 构建 Bottleneck ---
        self.bottom_block = ConvBlock(bottleneck_in, bottleneck_out, norm_layer=norm_layer)

        # --- 构建 Decoder 1: Depth Branch ---
        self.depth_upblocks = nn.ModuleList()
        current_ch = bottleneck_out # 初始为1024
        
        for i in range(self.n_layers):
            # 倒序获取Skip connection的通道数
            skip_ch = self.enc_channels[-(i+1)] 
            out_ch = self.depth_channels[i]
            
            # 输入通道 = 上一层输出 + Skip通道
            block = UpsampleBlock(current_ch + skip_ch, out_ch, norm_layer)
            self.depth_upblocks.append(block)
            current_ch = out_ch 

        # Depth 输出层: 16通道 -> 1通道
        self.depth_final = nn.Conv2d(16, 1, kernel_size=1)

        # --- 构建 Decoder 2: HS Image Branch ---
        self.hs_upblocks = nn.ModuleList()
        current_ch = bottleneck_out # 重置为1024
        
        for i in range(self.n_layers):
            skip_ch = self.enc_channels[-(i+1)]
            out_ch = self.hs_channels[i]
            
            block = UpsampleBlock(current_ch + skip_ch, out_ch, norm_layer)
            self.hs_upblocks.append(block)
            current_ch = out_ch

        # HS 输出层: 32通道 -> 25通道
        self.hs_final = nn.Conv2d(32, 29, kernel_size=1)

        # --- 权重初始化 ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # --- Encoder Path ---
        features = []
        for i in range(self.n_layers):
            # DownsampleBlock返回: 下一层输入x, Skip Connection y
            x, y = self.downblocks[i](x)
            features.append(y)
        
        # Bottleneck
        b = self.bottom_block(x)
        
        # --- Decoder Path 1: Depth ---
        d = b
        for i in range(self.n_layers):
            # 倒序取出对应的Encoder特征
            skip_feat = features[-(i+1)]
            d = self.depth_upblocks[i](d, skip_feat)
        
        depth_out = self.depth_final(d)

        # --- Decoder Path 2: HS Image ---
        h = b
        for i in range(self.n_layers):
            skip_feat = features[-(i+1)]
            h = self.hs_upblocks[i](h, skip_feat)
        
        # 直接输出解码器结果，不加残差
        hs_out = self.hs_final(h)

        return depth_out, hs_out

# ================= 测试代码 =================
if __name__ == "__main__":
    # 实例化模型 (不需要传入 crf_weights)
    model = DualHeadUNet()
    
    # 打印参数量信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    # 输入测试 [Batch, 3, 512, 512]
    x = torch.randn(1, 3, 512, 512)
    depth, hs = model(x)
    
    print("-" * 30)
    print(f"Input: {x.shape}")
    print(f"Depth Output: {depth.shape} (Expected: [1, 1, 512, 512])")
    print(f"HS Output:    {hs.shape}    (Expected: [1, 25, 512, 512])")
    print("-" * 30)