# model.py
import torch, torchvision
from torch import nn

def conv3x3(i, o): return nn.Conv2d(i, o, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(conv3x3(i, o), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, i, m, o):
        super().__init__()
        self.block = nn.Sequential(
            ConvRelu(i, m),
            nn.ConvTranspose2d(m, o, 3, 2, 1, 1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class UNet11_4ch(nn.Module):
    """UNet-11，首層改為 4→64，RGB 權重沿用並為第 4 通道初始化均值"""
    def __init__(self, num_filters=32, pretrained=True):
        super().__init__()
        enc = torchvision.models.vgg11(pretrained=pretrained).features
        old = enc[0]                                   # 3→64
        new = nn.Conv2d(4, 64, 3, padding=1)           # ★★ 改 4→64
        if pretrained:
            with torch.no_grad():
                new.weight[:, :3] = old.weight         # RGB 權重複製
                new.weight[:, 3:4] = old.weight.mean(1, keepdim=True)
                new.bias = old.bias
        enc[0] = new
        self.encoder = enc

        self.relu, self.pool = nn.ReLU(inplace=True), nn.MaxPool2d(2,2)
        self.conv1, self.conv2 = enc[0], enc[3]
        self.conv3s, self.conv3 = enc[6], enc[8]
        self.conv4s, self.conv4 = enc[11], enc[13]
        self.conv5s, self.conv5 = enc[16], enc[18]

        nf = num_filters
        self.center = DecoderBlock(nf*8*2, nf*8*2, nf*8)
        self.dec5   = DecoderBlock(nf*(16+8), nf*8*2, nf*8)
        self.dec4   = DecoderBlock(nf*(16+8), nf*8*2, nf*4)
        self.dec3   = DecoderBlock(nf*(8+4),  nf*4*2, nf*2)
        self.dec2   = DecoderBlock(nf*(4+2),  nf*2*2, nf)
        self.dec1   = ConvRelu(nf*(2+1), nf)
        self.final  = nn.Conv2d(nf, 1, 1)

    def forward(self, x):
        c1 = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(self.pool(c1)))
        c3s= self.relu(self.conv3s(self.pool(c2)))
        c3 = self.relu(self.conv3(c3s))
        c4s= self.relu(self.conv4s(self.pool(c3)))
        c4 = self.relu(self.conv4(c4s))
        c5s= self.relu(self.conv5s(self.pool(c4)))
        c5 = self.relu(self.conv5(c5s))

        ctr = self.center(self.pool(c5))
        d5  = self.dec5(torch.cat([ctr, c5], 1))
        d4  = self.dec4(torch.cat([d5,  c4], 1))
        d3  = self.dec3(torch.cat([d4,  c3], 1))
        d2  = self.dec2(torch.cat([d3,  c2], 1))
        d1  = self.dec1(torch.cat([d2,  c1], 1))
        return self.final(d1)