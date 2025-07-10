import torch
import torchvision
from torch import nn


def conv3x3(i: int, o: int) -> nn.Conv2d:
    """3×3 convolution with padding=1 (same spatial size)."""
    return nn.Conv2d(i, o, kernel_size=3, padding=1, bias=False)


class ConvGelu(nn.Module):
    """Conv → GELU helper block (replacing Conv+ReLU)."""

    def __init__(self, i: int, o: int):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(i, o),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Decoder up‑sampling block: (Conv→GELU) ×1  +  2× upsample via transpose‑conv."""

    def __init__(self, i: int, m: int, o: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvGelu(i, m),
            nn.ConvTranspose2d(m, o, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class InvertedBottleneck(nn.Module):
    """MobileNet‑v2 style inverted residual bottleneck.

    Args:
        ch (int): input/output channels.
        exp (int): expansion factor. Default 4 (i.e. ch → ch*exp → ch).
        use_se (bool): whether to use Squeeze-and-Excitation.
    """

    def __init__(self, ch: int, exp: int = 4, use_se: bool = False):
        super().__init__()
        mid = ch * exp
        
        self.expand = nn.Sequential(
            nn.Conv2d(ch, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )
        
        # Squeeze-and-Excitation block (optional)
        self.se = None
        if use_se:
            self.se = SqueezeExcitation(mid)
        
        self.project = nn.Sequential(
            nn.Conv2d(mid, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        if self.se is not None:
            x = self.se(x)
        x = self.project(x)
        return residual + x


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class MultiScaleBottleneck(nn.Module):
    """Multi-scale bottleneck with different dilation rates."""
    
    def __init__(self, ch: int, exp: int = 4):
        super().__init__()
        mid = ch * exp
        
        # Different dilation rates for multi-scale feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(ch, mid // 4, 1, bias=False),
            nn.BatchNorm2d(mid // 4),
            nn.GELU(),
            nn.Conv2d(mid // 4, mid // 4, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid // 4),
            nn.GELU(),
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(ch, mid // 4, 1, bias=False),
            nn.BatchNorm2d(mid // 4),
            nn.GELU(),
            nn.Conv2d(mid // 4, mid // 4, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid // 4),
            nn.GELU(),
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(ch, mid // 4, 1, bias=False),
            nn.BatchNorm2d(mid // 4),
            nn.GELU(),
            nn.Conv2d(mid // 4, mid // 4, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid // 4),
            nn.GELU(),
        )
        
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid // 4, 1, bias=True),  # 使用 bias=True，避免 BatchNorm
            nn.GELU(),
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(mid, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Upsample branch4 to match spatial dimensions
        b4 = nn.functional.interpolate(b4, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate all branches
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.project(concat)
        
        return x + out


class EnhancedUNet11_4ch(nn.Module):
    """Enhanced UNet‑11 with multiple bottleneck improvements."""

    def __init__(self, num_filters: int = 32, pretrained: bool = True, bottleneck_type: str = 'multi'):
        super().__init__()
        
        # ─────── Encoder (VGG11) ────────────────────────────────────────────
        enc = torchvision.models.vgg11(pretrained=pretrained).features

        # First conv: 4→64 (copy RGB weights and set 4th channel to mean)
        old_conv1 = enc[0]  # 3→64
        new_conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=False)
        if pretrained:
            with torch.no_grad():
                new_conv1.weight[:, :3] = old_conv1.weight  # copy RGB
                new_conv1.weight[:, 3:4] = old_conv1.weight.mean(dim=1, keepdim=True)
        enc[0] = new_conv1
        self.encoder = enc

        # Shortcut handles
        self.gelu = nn.GELU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1, self.conv2 = enc[0], enc[3]
        self.conv3s, self.conv3 = enc[6], enc[8]
        self.conv4s, self.conv4 = enc[11], enc[13]
        self.conv5s, self.conv5 = enc[16], enc[18]

        # ─────── Enhanced Decoder with Bottlenecks ──────────────────────────
        nf = num_filters
        
        # 確保通道數匹配
        self.center = DecoderBlock(nf * 16, nf * 8 * 2, nf * 8)  # 512 -> 256 -> 256
        
        # Add bottlenecks in decoder layers
        self.dec5 = DecoderBlock(nf * (8 + 16), nf * 8 * 2, nf * 8)  # 768 -> 512 -> 256
        self.dec5_bottleneck = self._create_bottleneck(nf * 8, bottleneck_type)
        
        self.dec4 = DecoderBlock(nf * (8 + 16), nf * 8 * 2, nf * 4)  # 768 -> 512 -> 128
        self.dec4_bottleneck = self._create_bottleneck(nf * 4, bottleneck_type)
        
        self.dec3 = DecoderBlock(nf * (4 + 8), nf * 4 * 2, nf * 2)  # 384 -> 256 -> 64
        self.dec3_bottleneck = self._create_bottleneck(nf * 2, bottleneck_type)
        
        self.dec2 = DecoderBlock(nf * (2 + 4), nf * 2 * 2, nf)  # 192 -> 128 -> 32
        self.dec2_bottleneck = self._create_bottleneck(nf, bottleneck_type)
        
        self.dec1 = ConvGelu(nf * (1 + 2), nf)  # 96 -> 32

        # ─────── Multiple Final Bottlenecks ──────────────────────────────────
        self.final_bottlenecks = nn.ModuleList([
            self._create_bottleneck(nf, bottleneck_type) for _ in range(3)
        ])
        
        # Final convolution
        self.final = nn.Conv2d(nf, 1, kernel_size=1)

    def _create_bottleneck(self, channels: int, bottleneck_type: str):
        """Create bottleneck based on type."""
        if bottleneck_type == 'multi':
            return MultiScaleBottleneck(channels, exp=4)
        elif bottleneck_type == 'se':
            return InvertedBottleneck(channels, exp=4, use_se=True)
        else:  # 'standard'
            return InvertedBottleneck(channels, exp=4, use_se=False)

    def _enc_forward(self, x):
        c1 = self.gelu(self.conv1(x))
        c2 = self.gelu(self.conv2(self.pool(c1)))
        c3s = self.gelu(self.conv3s(self.pool(c2)))
        c3 = self.gelu(self.conv3(c3s))
        c4s = self.gelu(self.conv4s(self.pool(c3)))
        c4 = self.gelu(self.conv4(c4s))
        c5s = self.gelu(self.conv5s(self.pool(c4)))
        c5 = self.gelu(self.conv5(c5s))
        return c1, c2, c3, c4, c5

    def forward(self, x):
        c1, c2, c3, c4, c5 = self._enc_forward(x)
        ctr = self.center(self.pool(c5))

        # Enhanced decoder with bottlenecks
        d5 = self.dec5(torch.cat([ctr, c5], dim=1))
        d5 = self.dec5_bottleneck(d5)
        
        d4 = self.dec4(torch.cat([d5, c4], dim=1))
        d4 = self.dec4_bottleneck(d4)
        
        d3 = self.dec3(torch.cat([d4, c3], dim=1))
        d3 = self.dec3_bottleneck(d3)
        
        d2 = self.dec2(torch.cat([d3, c2], dim=1))
        d2 = self.dec2_bottleneck(d2)
        
        d1 = self.dec1(torch.cat([d2, c1], dim=1))

        # Apply multiple final bottlenecks
        out = d1
        for bottleneck in self.final_bottlenecks:
            out = bottleneck(out)
        
        return self.final(out)


# 使用示例
if __name__ == "__main__":
    # 三種不同的瓶頸配置
    model_multi = EnhancedUNet11_4ch(num_filters=32, bottleneck_type='multi')
    model_se = EnhancedUNet11_4ch(num_filters=32, bottleneck_type='se')
    model_standard = EnhancedUNet11_4ch(num_filters=32, bottleneck_type='standard')
    
    # 測試輸入
    x = torch.randn(1, 4, 256, 256)
    
    with torch.no_grad():
        out_multi = model_multi(x)
        out_se = model_se(x)
        out_standard = model_standard(x)
    
    print(f"Multi-scale output shape: {out_multi.shape}")
    print(f"SE output shape: {out_se.shape}")
    print(f"Standard output shape: {out_standard.shape}")