from model.resnet_custom import resnet18_baseline, resnet50_baseline, resnet34_baseline
import torch.nn as nn
import torch.nn.functional as F
import torch
from model.resnet_custom import resnet18_baseline, resnet50_baseline


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inconv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x, withAct=False):
        x = self.conv(x)
        if withAct:
            x = self.act(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = Inconv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = Outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feature = self.up4(x, x1)
        x = self.outc(feature)
        return feature, x



class ResNet_UNet(nn.Module):

    def __init__(self, network='resnet18', pretrain=True, nClass=2, frozen=False):
        super(ResNet_UNet, self).__init__()
        self.frozen = frozen
        if network == 'resnet18':
            self.frontNet = resnet18_baseline(pretrain)
            inChn = 256
        elif network == 'resnet50':
            self.frontNet = resnet50_baseline(pretrain)
            inChn = 1024
        elif network == 'resnet34':
            self.frontNet = resnet34_baseline(pretrain)
            inChn = 256

        self.mUnet = UNet(inChn, nClass)

        self.inConv = nn.Conv2d(inChn, inChn, 1, bias=False)

    def forward(self, x):

        if self.frozen:
            with torch.no_grad():
                feature_0 = self.frontNet(x)
        else:
            feature_0 = self.frontNet(x)

        feature_0 = self.inConv(feature_0)
        feature_1, x = self.mUnet(feature_0)

        return x, feature_0, feature_1



class ResNet_UNet_PretrainFeature(nn.Module):

    def __init__(self, network='resnet18', pretrain=True, nClass=2):
        super(ResNet_UNet_PretrainFeature, self).__init__()
        #self.frozen = frozen
        if network == 'resnet18':
            #self.frontNet = resnet18_baseline(pretrain)
            inChn = 256
        elif network == 'resnet50':
            #self.frontNet = resnet50_baseline(pretrain)
            inChn = 1024
        elif network == 'resnet34':
            #self.frontNet = resnet34_baseline(pretrain)
            inChn = 256

        self.mUnet = UNet(inChn, nClass)

        self.inConv = nn.Conv2d(inChn, inChn, 1, bias=False)

    def forward(self, x):

        feature_0 = self.inConv(x)
        feature_1, x = self.mUnet(feature_0)

        return x, feature_0, feature_1






