import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgan.layers import SpectralNorm2d
import enum

from ssim import msssim

#最有效
NUM_BANDS = 1
PATCH_SIZE = 256
SCALE_FACTOR = 1

class RFLU(nn.Module):
    def __init__(self):
        super(RFLU, self).__init__()

    def forward(self, c1,c2,f2):
        '''
        w1 = torch.where(c1 > 0, torch.full_like(c1, 1),torch.full_like(c1, 0))  # 目标时刻大于0，但也有可能是噪音、益光
        z = f2 * w1  # 此时目标时间小于等于0的，都变为了0
        w2=torch.where(c2 < 0, torch.full_like(c2,0), torch.full_like(c2,1))#参考时间小于0，f2舍弃原特征
        z= z * w2
        w3=torch.where(c2 < 0, c1 , torch.full_like(c2,0))#参考时间小于0，f2舍弃原特征
        #w3=w3*0.1
        z = z + w3'''
        z = torch.where(c1 >= 0, torch.where(c2 >= 0, f2 , c1),torch.full_like(c1, 0))
            
        return z

class Sampling(enum.Enum):
    UpSampling = enum.auto()
    DownSampling = enum.auto()
    Identity = enum.auto()

class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return F.interpolate(inputs, scale_factor=self.scale_factor)

def calc_mean_std(features):
    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features, style_features):
    """
    Adaptive Instance Normalization
    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
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
        #_prediction, _target = self.encoder(prediction), self.encoder(target)
        loss = (self.alpha * F.mse_loss(prediction, target) +
                #self.gamma * (1.0 - torch.mean(F.cosine_similarity(prediction, target, 1))) +#光谱损失
                self.beta * (1.0 - msssim(prediction, target, normalize=True)))
        return loss


class Conv3X3NoPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3NoPadding, self).__init__(in_channels, out_channels, 3, stride=stride, padding=1)


class Conv3X3WithPadding(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3WithPadding, self).__init__(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        )


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, sampling=None):
        layers = []

        if sampling == Sampling.DownSampling:
            layers.append(Conv3X3WithPadding(in_channels, out_channels, 2))
        else:
            if sampling == Sampling.UpSampling:
                layers.append(Upsample(2))
            layers.append(Conv3X3WithPadding(in_channels, out_channels))

        layers.append(nn.ReLU(inplace=True))
        super(ConvBlock, self).__init__(*layers)

#高分辨率残差块
class ResidulBlockWtihSwitchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, sampling=None):
        super(ResidulBlockWtihSwitchNorm, self).__init__()
        channels = min(in_channels, out_channels)
        residual = [
            #SwitchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            Conv3X3WithPadding(in_channels, channels),
            #nn.Dropout(0.75),
            #SwitchNorm2d(channels),
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
        l1_list = self.cencoder(inputs[2])  # 目标高
        
        c21 = self.conv1(inputs[1])
        
        ad1=adain(l1_list[0], c1_list[0])
        #rf1=self.rflu(c1_list[0], c21, ad1)
        c22 = self.conv2(ad1+c21)
        
        ad2=adain(l1_list[1], c1_list[1])
        #rf2=self.rflu(c1_list[1], c22, ad2)
        c23 = self.conv3(ad2+c22)
        
        ad3=adain(l1_list[2], c1_list[2])
        #rf3=self.rflu(c1_list[2], c23, ad3)
        c24 = self.conv4(ad3+c23)
        
        ad4=adain(l1_list[3], c1_list[3])
        rf4=self.rflu(c1_list[3], c24, ad4)
            
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

#自动编码器
""""""
class AutoEncoder(nn.Module):
    def __init__(self, in_channels=NUM_BANDS, out_channels=NUM_BANDS):
        super(AutoEncoder, self).__init__()
        channels = (16, 32, 64, 128)
        self.conv1 = ConvBlock(in_channels, channels[0])
        self.conv2 = ConvBlock(channels[0], channels[1], Sampling.DownSampling)
        self.conv3 = ConvBlock(channels[1], channels[2], Sampling.DownSampling)
        self.conv4 = ConvBlock(channels[2], channels[3], Sampling.DownSampling)
        self.conv5 = ConvBlock(channels[3], channels[2], Sampling.UpSampling)
        self.conv6 = ConvBlock(channels[2] * 2, channels[1], Sampling.UpSampling)
        self.conv7 = ConvBlock(channels[1] * 2, channels[0], Sampling.UpSampling)
        self.conv8 = nn.Conv2d(channels[0] * 2, out_channels, 1)

    def forward(self, inputs):
        l1 = self.conv1(inputs)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        l5 = self.conv5(l4)
        l6 = self.conv6(torch.cat((l3, l5), 1))
        l7 = self.conv7(torch.cat((l2, l6), 1))
        out = self.conv8(torch.cat((l1, l7), 1))
        return out

#粗分辨率残差块
class Resblock(nn.Module):
    def __init__(self, channels):
        super(Resblock, self).__init__()
        self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.PReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.Dropout(0.75)
            )

    def forward(self, x):
        residual = self.residual(x)
        return x + residual

#高分辨率
# class FEncoder(nn.Sequential):
#     def __init__(self):
#         channels = [16, 32, 64, 128]
#         super(FEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             ResidulBlockWtihSwitchNorm(NUM_BANDS, channels[0]),
#             ResidulBlockWtihSwitchNorm(channels[0], channels[1]),
#             ResidulBlockWtihSwitchNorm(channels[1], channels[2]),
#             ResidulBlockWtihSwitchNorm(channels[2], channels[3])
#         )
#     def forward(self, x):
#         residual = self.encoder(x)
#         return residual


#粗分辨率
class REncoder(nn.Sequential):
    def __init__(self):
        #channels = [NUM_BANDS * 3, 32, 64, 128]
        super(REncoder, self).__init__()
        # 粗分辨率图像
        self.conv1 = nn.Sequential(
            nn.Conv2d(NUM_BANDS, 64, kernel_size=5, padding=2),
            # nn.Conv2d(3,64,kernel_size=9,padding=4,padding_mode='reflect',stride=1)
            nn.PReLU()
            
        )
        trunk = []
        for i in range(16):
            trunk.append(Resblock(64))
        self.trunk = nn.Sequential(*trunk)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv3 = nn.Conv2d(64, 128, 1)

    def forward(self, inputs):
        block1 = self.conv1(inputs)
        block2 = self.trunk(block1)
        block3 = self.conv3(block1) + self.conv2(block2)
        return block3

# class CEncoder(nn.Sequential):
#     def __init__(self):
#         channels = [16, 32, 64, 128]
#         super(CEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             ResidulBlockWtihSwitchNorm(NUM_BANDS*2, channels[0]),
#             ResidulBlockWtihSwitchNorm(channels[0], channels[1]),
#             ResidulBlockWtihSwitchNorm(channels[1], channels[2]),
#             ResidulBlockWtihSwitchNorm(channels[2], channels[3])
#         )
#     def forward(self, x):
#         residual = self.encoder(x)
#         return residual

# class SFFusion(nn.Module):
#     def __init__(self, in_channels=NUM_BANDS, out_channels=NUM_BANDS):
#         super(SFFusion, self).__init__()    
        
#         self.fencoder = FEncoder()
#         channels = (16, 32, 64, 128)
#         self.rflu_list = nn.ModuleList()
#         for i in range(len(channels)):#注意力机制
#             self.rflu_list.append(RFLU().cuda())

#         self.conv1 = nn.Sequential(
#             ResidulBlock(channels[3], channels[3]),
#         )
#         self.conv2 = nn.Sequential(
#             ResidulBlock(channels[3], channels[2]),
#         )
#         self.conv3 = nn.Sequential(
#             ResidulBlock(channels[2]*2, channels[1]),
#         )
#         self.conv4 = nn.Sequential(
#             ResidulBlock(channels[1]*2, channels[0]),
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(channels[0]*2, out_channels, 1)
#         )

#     def forward(self, inputs):

#         c1_list = self.fencoder(inputs[0])  # 目标低
#         c2_list = self.fencoder(inputs[2])  # 参考低
#         f2_list = self.fencoder(inputs[1])  # 参考高
#         # c = self.cencoder(torch.cat((inputs[0], inputs[2]),1))
#         # prev=rflu(c,f2)
#         ad_list = []
#         for c1, f2 in zip(c1_list, f2_list):
#             ad_list.append(adain(f2, c1))#计算adain

#         rf_list = []
#         for rf, ad, c1, c2 in zip(self.rflu_list, ad_list, c1_list, c2_list):
#             rf_out = rf(c1, c2, ad)
#             rf_list.append(rf_out)
#         l4 = self.conv1(rf_list[3])
#         l4 = self.conv2(rf_list[3])
#         l3 = self.conv3(torch.cat((rf_list[2], l4),dim=1))
#         l2 = self.conv4(torch.cat((rf_list[1], l3),dim=1))
#         l1 = self.conv5(torch.cat((rf_list[0], l2),dim=1))

#         return l1
    
    
#生成器网络
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
        
        
        #prev_diff = self.cencoder(torch.cat((inputs[0], inputs[2]),1))
        #return self.decoder(adain(self.fencoder(inputs[1]) , self.rencoder(inputs[0])) + prev_diff )