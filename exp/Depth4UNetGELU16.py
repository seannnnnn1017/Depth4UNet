# model.py
import torch, torchvision
from torch import nn

def conv3x3(i, o): return nn.Conv2d(i, o, 3, padding=1)

class ConvGelu(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(conv3x3(i, o), nn.GELU())
    def forward(self, x): return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, i, m, o):
        super().__init__()
        self.block = nn.Sequential(
            ConvGelu(i, m),
            nn.ConvTranspose2d(m, o, 3, 2, 1, 1),
            nn.GELU()
        )
    def forward(self, x): return self.block(x)

class UNet16_4ch(nn.Module):
    """UNet-16，首層改為 4→64，RGB 權重沿用並為第 4 通道初始化均值"""
    def __init__(self, num_filters=32, pretrained=True):
        super().__init__()
        enc = torchvision.models.vgg16(pretrained=pretrained).features
        old = enc[0]                                   # 3→64
        new = nn.Conv2d(4, 64, 3, padding=1)           # ★★ 改 4→64
        if pretrained:
            with torch.no_grad():
                new.weight[:, :3] = old.weight         # RGB 權重複製
                new.weight[:, 3:4] = old.weight.mean(1, keepdim=True)
                new.bias = old.bias
        enc[0] = new
        self.encoder = enc

        self.gelu, self.pool = nn.GELU(), nn.MaxPool2d(2,2)
        
        # VGG16 layer mapping
        self.conv1, self.conv1_2 = enc[0], enc[2]      # 第1組：conv + conv
        self.conv2, self.conv2_2 = enc[5], enc[7]      # 第2組：conv + conv
        self.conv3, self.conv3_2, self.conv3_3 = enc[10], enc[12], enc[14]  # 第3組：conv + conv + conv
        self.conv4, self.conv4_2, self.conv4_3 = enc[17], enc[19], enc[21]  # 第4組：conv + conv + conv
        self.conv5, self.conv5_2, self.conv5_3 = enc[24], enc[26], enc[28]  # 第5組：conv + conv + conv

        nf = num_filters
        self.center = DecoderBlock(nf*8*2, nf*8*2, nf*8)
        self.dec5   = DecoderBlock(nf*(16+8), nf*8*2, nf*8)
        self.dec4   = DecoderBlock(nf*(16+8), nf*8*2, nf*4)
        self.dec3   = DecoderBlock(nf*(8+4),  nf*4*2, nf*2)
        self.dec2   = DecoderBlock(nf*(4+2),  nf*2*2, nf)
        self.dec1   = ConvGelu(nf*(2+1), nf)
        self.final  = nn.Conv2d(nf, 1, 1)

    def forward(self, x):
        # VGG16 encoder with GELU activation
        c1_1 = self.gelu(self.conv1(x))
        c1_2 = self.gelu(self.conv1_2(c1_1))
        
        c2_1 = self.gelu(self.conv2(self.pool(c1_2)))
        c2_2 = self.gelu(self.conv2_2(c2_1))
        
        c3_1 = self.gelu(self.conv3(self.pool(c2_2)))
        c3_2 = self.gelu(self.conv3_2(c3_1))
        c3_3 = self.gelu(self.conv3_3(c3_2))
        
        c4_1 = self.gelu(self.conv4(self.pool(c3_3)))
        c4_2 = self.gelu(self.conv4_2(c4_1))
        c4_3 = self.gelu(self.conv4_3(c4_2))
        
        c5_1 = self.gelu(self.conv5(self.pool(c4_3)))
        c5_2 = self.gelu(self.conv5_2(c5_1))
        c5_3 = self.gelu(self.conv5_3(c5_2))

        # Decoder with skip connections
        ctr = self.center(self.pool(c5_3))
        d5  = self.dec5(torch.cat([ctr, c5_3], 1))
        d4  = self.dec4(torch.cat([d5,  c4_3], 1))
        d3  = self.dec3(torch.cat([d4,  c3_3], 1))
        d2  = self.dec2(torch.cat([d3,  c2_2], 1))
        d1  = self.dec1(torch.cat([d2,  c1_2], 1))
        return self.final(d1)