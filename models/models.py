import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import models


def conv3x3(in_: int, out: int) -> nn.Module:
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int) -> None:
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, middle_channels: int, out_channels: int
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_filters: int = 32, pretrained: bool = False) -> None:
        """

        Args:
            num_filters:
            pretrained:
                False - no pre-trained network is used
                True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(
            num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8
        )
        self.dec5 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8
        )
        self.dec4 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4
        )
        self.dec3 = DecoderBlock(
            num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2
        )
        self.dec2 = DecoderBlock(
            num_filters * (4 + 2), num_filters * 2 * 2, num_filters
        )
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


class Interpolate(nn.Module):
    def __init__(
        self,
        size: int = None,
        scale_factor: int = None,
        mode: str = "nearest",
        align_corners: bool = False,
    ):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.interp(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class DecoderBlockV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        is_deconv: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels, out_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode="bilinear"),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet16(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        num_filters: int = 32,
        pretrained: bool = False,
        is_deconv: bool = False,
    ):
        """

        Args:
            num_classes:
            num_filters:
            pretrained:
                False - no pre-trained network used
                True - encoder pre-trained with VGG16
            is_deconv:
                False: bilinear interpolation is used in decoder
                True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            self.encoder[0], self.relu, self.encoder[2], self.relu
        )

        self.conv2 = nn.Sequential(
            self.encoder[5], self.relu, self.encoder[7], self.relu
        )

        self.conv3 = nn.Sequential(
            self.encoder[10],
            self.relu,
            self.encoder[12],
            self.relu,
            self.encoder[14],
            self.relu,
        )

        self.conv4 = nn.Sequential(
            self.encoder[17],
            self.relu,
            self.encoder[19],
            self.relu,
            self.encoder[21],
            self.relu,
        )

        self.conv5 = nn.Sequential(
            self.encoder[24],
            self.relu,
            self.encoder[26],
            self.relu,
            self.encoder[28],
            self.relu,
        )

        self.center = DecoderBlockV2(
            512, num_filters * 8 * 2, num_filters * 8, is_deconv
        )

        self.dec5 = DecoderBlockV2(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv
        )
        self.dec4 = DecoderBlockV2(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv
        )
        self.dec3 = DecoderBlockV2(
            256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv
        )
        self.dec2 = DecoderBlockV2(
            128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv
        )
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)



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


