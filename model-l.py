import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgan.layers import SpectralNorm2d
import enum

from ssim import msssim

NUM_BANDS = 1
PATCH_SIZE = 256
SCALE_FACTOR = 1

class RFLU(nn.Module):
    def __init__(self):
        super(RFLU, self).__init__()

    def forward(self, c1,c2,f2):
        w1 = torch.where(c1 > 0, torch.full_like(c1, 1),torch.full_like(c1, 0))  # 目标时刻大于0，但也有可能是噪音、益光
        z = f2 * w1  # 此时目标时间小于等于0的，都变为了0
        w2=torch.where(c2 < 0, torch.full_like(c2,0), torch.full_like(c2,1))#参考时间小于0，f2舍弃原特征
        z= z * w2
        w3=torch.where(c2 < 0, c1 , torch.full_like(c2,0))#参考时间小于0，f2舍弃原特征
        #w3=w3*0.1#参数
        z = z + w3
            
        return z

class Sampling(enum.Enum):
    UpSampling = enum.auto()
    DownSampling = enum.auto()
    Identity = enum.auto()


def calc_mean_std(features):
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features, style_features):
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features


#图像损失
class ReconstructionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(ReconstructionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, prediction, target):
        loss = (self.alpha * F.mse_loss(prediction, target) +
                self.beta * (1.0 - msssim(prediction, target, normalize=True)))
        return loss


class Conv3X3WithPadding(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3WithPadding, self).__init__(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        )

        
#高分辨率残差块
class ResidulBlockWtihSwitchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, sampling=None):
        super(ResidulBlockWtihSwitchNorm, self).__init__()
        channels = min(in_channels, out_channels)
        residual = [
            nn.LeakyReLU(inplace=True),
            Conv3X3WithPadding(in_channels, channels),
            #nn.Dropout(0.75),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 1),
            #nn.Dropout(0.75)
        ]
        transform = [
            nn.Conv2d(channels, out_channels, 1)
        ]

        self.residual = nn.Sequential(*residual)
        self.transform = nn.Sequential(*transform)

    def forward(self, inputs):
        trunk = self.residual(inputs)
        lateral = self.transform(inputs)
        return trunk + lateral

    
class CEncoder(nn.Sequential):
    def __init__(self):
        super(CEncoder, self).__init__()
        channels = (16, 32, 64, 128)
        self.conv1 = nn.Sequential(
            ResidulBlockWtihSwitchNorm(NUM_BANDS, channels[0]),
        )
        self.conv2 = nn.Sequential(
            ResidulBlockWtihSwitchNorm(channels[0], channels[1]),
        )
        self.conv3 = nn.Sequential(
            ResidulBlockWtihSwitchNorm(channels[1], channels[2]),
        )
        self.conv4 = nn.Sequential(
            ResidulBlockWtihSwitchNorm(channels[2], channels[3]),
        )

    def forward(self, inputs):
        l1 = self.conv1(inputs)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        return [l1, l2, l3, l4]   

    
class FEncoder(nn.Sequential):
    def __init__(self):
        super(FEncoder, self).__init__()
        channels = (16, 32, 64, 128)
        self.cencoder = CEncoder()
        self.conv1 = nn.Sequential(
            ResidulBlockWtihSwitchNorm(NUM_BANDS, channels[0]),
        )
        self.conv2 = nn.Sequential(
            ResidulBlockWtihSwitchNorm(channels[0], channels[1]),
        )
        self.conv3 = nn.Sequential(
            ResidulBlockWtihSwitchNorm(channels[1], channels[2]),
        )
        self.conv4 = nn.Sequential(
            ResidulBlockWtihSwitchNorm(channels[2], channels[3]),
        )
        self.rflu = RFLU()

    def forward(self, inputs):
        
        c1_list = self.cencoder(inputs[0])  # 目标低
        c2_list = self.cencoder(inputs[2])  # 参考低
        
        l1 = self.conv1(inputs[1])
        
        ad1=adain(l1, c1_list[0])
        rf1=self.rflu(c1_list[0], c2_list[0], ad1)
        l2 = self.conv2(rf1+l1)
        
        ad2=adain(l2, c1_list[1])
        rf2=self.rflu(c1_list[1], c2_list[1], ad2)
        l3 = self.conv3(rf2+l2)
        
        ad3=adain(l3, c1_list[2])
        rf3=self.rflu(c1_list[2], c2_list[2], ad3)
        l4 = self.conv4(rf3+l3)
        
        ad4=adain(l4, c1_list[3])
        rf4=self.rflu(c1_list[3], c2_list[3], ad4)
            
        return rf4

    
#decoder残差块
class ResidulBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sampling=None):
        super(ResidulBlock, self).__init__()
        channels = min(in_channels, out_channels)
        residual = [
            Conv3X3WithPadding(in_channels, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 1)
        ]
        transform = [nn.Conv2d(in_channels, out_channels, 1)]

        self.residual = nn.Sequential(*residual)
        self.transform = transform[0] if len(transform) == 1 else nn.Sequential(*transform)

    def forward(self, inputs):
        trunk = self.residual(inputs)
        lateral = self.transform(inputs)
        return trunk + lateral

    
#融合网络
class SFFusion(nn.Module):
    def __init__(self, in_channels=NUM_BANDS, out_channels=NUM_BANDS):
        channels = (16, 32, 64, 128)
        super(SFFusion, self).__init__()
        #粗分辨率图像
        self.fencoder = FEncoder()
        self.decoder = nn.Sequential(
            ResidulBlock(channels[3], channels[3]),
            ResidulBlock(channels[3], channels[2]),
            ResidulBlock(channels[2], channels[1]),
            ResidulBlock(channels[1], channels[0]),
            nn.Conv2d(channels[0], out_channels, 1)
        )

    def forward(self, inputs):
        f2 = self.fencoder(inputs)

        return self.decoder(f2)