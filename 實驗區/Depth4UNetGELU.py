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
    """Decoder up‑sampling block: (Conv→GELU) ×1  +  2× upsample via transpose‑conv."""

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
    """

    def __init__(self, ch: int, exp: int = 4):
        super().__init__()
        mid = ch * exp
        self.block = nn.Sequential(
            # 1×1 point‑wise expansion
            nn.Conv2d(ch, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            # 3×3 depth‑wise conv
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            # 1×1 point‑wise projection back
            nn.Conv2d(mid, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)  # residual connection


class UNet11_4ch(nn.Module):
    """UNet‑11 backbone with 4‑channel input, GELU activations, and an extra inverted bottleneck before the final head."""

    def __init__(self, num_filters: int = 32, pretrained: bool = True):
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
        self.encoder = enc  # keep as torch.nn.Sequential

        # Shortcut handles
        self.gelu = nn.GELU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1, self.conv2 = enc[0], enc[3]
        self.conv3s, self.conv3 = enc[6], enc[8]
        self.conv4s, self.conv4 = enc[11], enc[13]
        self.conv5s, self.conv5 = enc[16], enc[18]

        # ─────── Decoder ──────────────────────────────────────────────────
        nf = num_filters
        self.center = DecoderBlock(nf * 8 * 2, nf * 8 * 2, nf * 8)
        self.dec5 = DecoderBlock(nf * (16 + 8), nf * 8 * 2, nf * 8)
        self.dec4 = DecoderBlock(nf * (16 + 8), nf * 8 * 2, nf * 4)
        self.dec3 = DecoderBlock(nf * (8 + 4), nf * 4 * 2, nf * 2)
        self.dec2 = DecoderBlock(nf * (4 + 2), nf * 2 * 2, nf)
        self.dec1 = ConvGelu(nf * (2 + 1), nf)

        # ─────── Inverted Bottleneck  +  Final head ────────────────────────
        self.inv = InvertedBottleneck(nf, exp=4)
        self.final = nn.Conv2d(nf, 1, kernel_size=1)

    # ---------------------------------------------------------------------
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

        d5 = self.dec5(torch.cat([ctr, c5], dim=1))
        d4 = self.dec4(torch.cat([d5, c4], dim=1))
        d3 = self.dec3(torch.cat([d4, c3], dim=1))
        d2 = self.dec2(torch.cat([d3, c2], dim=1))
        d1 = self.dec1(torch.cat([d2, c1], dim=1))

        out = self.inv(d1)  # apply inverted bottleneck residual block
        return self.final(out)


