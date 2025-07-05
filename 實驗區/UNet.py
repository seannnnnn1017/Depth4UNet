# model.py
import torch
from torch import nn

def conv3x3(i, o): 
    return nn.Conv2d(i, o, 3, padding=1)

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
      
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet_4ch(nn.Module):
  
    def __init__(self, n_channels=4, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
       
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
    
        self.outc = nn.Conv2d(64, n_classes, 1)
    
    def forward(self, x):
    
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
       
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        

        logits = self.outc(x)
        return logits


class UNet_4ch_Pretrained(nn.Module):

    def __init__(self, n_channels=4, n_classes=1, pretrained_3ch_weights=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
      
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
   
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
     
        self.outc = nn.Conv2d(64, n_classes, 1)
        
       
        if pretrained_3ch_weights is not None:
            self._init_4th_channel(pretrained_3ch_weights)
    
    def _init_4th_channel(self, pretrained_weights):
        """从3通道预训练权重初始化第4通道"""
       
        first_conv = self.inc.double_conv[0]
        if hasattr(pretrained_weights, 'inc.double_conv.0.weight'):
            pretrained_conv_weight = pretrained_weights['inc.double_conv.0.weight']
            pretrained_conv_bias = pretrained_weights['inc.double_conv.0.bias']
            
            with torch.no_grad():
     
                first_conv.weight[:, :3] = pretrained_conv_weight
      
                first_conv.weight[:, 3:4] = pretrained_conv_weight.mean(1, keepdim=True)
                first_conv.bias = pretrained_conv_bias
    
    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
 
        logits = self.outc(x)
        return logits