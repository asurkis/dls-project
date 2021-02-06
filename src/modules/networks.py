import torch
from torch import nn

class MyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # [ 3, 64, 128, 256, 512, 512, 512, 512, 512 ]

        self.downsamplers = [                                                                                                                          # 256 x 256 x   3
            nn.Sequential(nn.Conv2d(   3,  64, 4, stride=2, padding=1, padding_mode='replicate', bias=False ),                        nn.LeakyReLU()), # 128 x 128 x  64
            nn.Sequential(nn.Conv2d(  64, 128, 4, stride=2, padding=1, padding_mode='replicate', bias=False ), nn.BatchNorm2d( 128 ), nn.LeakyReLU()), #  64 x  64 x 128
            nn.Sequential(nn.Conv2d( 128, 256, 4, stride=2, padding=1, padding_mode='replicate', bias=False ), nn.BatchNorm2d( 256 ), nn.LeakyReLU()), #  32 x  32 x 256
            nn.Sequential(nn.Conv2d( 256, 512, 4, stride=2, padding=1, padding_mode='replicate', bias=False ), nn.BatchNorm2d( 512 ), nn.LeakyReLU()), #  16 x  16 x 512
            nn.Sequential(nn.Conv2d( 512, 512, 4, stride=2, padding=1, padding_mode='replicate', bias=False ), nn.BatchNorm2d( 512 ), nn.LeakyReLU()), #   8 x   8 x 512
            nn.Sequential(nn.Conv2d( 512, 512, 4, stride=2, padding=1, padding_mode='replicate', bias=False ), nn.BatchNorm2d( 512 ), nn.LeakyReLU()), #   4 x   4 x 512
            nn.Sequential(nn.Conv2d( 512, 512, 4, stride=2, padding=1, padding_mode='replicate', bias=False ), nn.BatchNorm2d( 512 ), nn.LeakyReLU()), #   2 x   2 x 512
            nn.Sequential(nn.Conv2d( 512, 512, 4, stride=2, padding=1, padding_mode='replicate', bias=False ), nn.BatchNorm2d( 512 ), nn.LeakyReLU()), #   1 x   1 x 512
        ]

        self.downsampler1 = self.downsamplers[0]
        self.downsampler2 = self.downsamplers[1]
        self.downsampler3 = self.downsamplers[2]
        self.downsampler4 = self.downsamplers[3]
        self.downsampler5 = self.downsamplers[4]
        self.downsampler6 = self.downsamplers[5]
        self.downsampler7 = self.downsamplers[6]
        self.downsampler8 = self.downsamplers[7]

        self.upsamplers = [                                                                                                         #                             1 x   1 x  512
            nn.Sequential(nn.ConvTranspose2d(  512, 512, 4, stride=2, padding=1 ), nn.BatchNorm2d( 512 ), nn.Dropout(), nn.ReLU()), #   2 x   2 x (512 + 512) =   2 x   2 x 1024
            nn.Sequential(nn.ConvTranspose2d( 1024, 512, 4, stride=2, padding=1 ), nn.BatchNorm2d( 512 ), nn.Dropout(), nn.ReLU()), #   4 x   4 x (512 + 512) =   4 x   4 x 1024
            nn.Sequential(nn.ConvTranspose2d( 1024, 512, 4, stride=2, padding=1 ), nn.BatchNorm2d( 512 ), nn.Dropout(), nn.ReLU()), #   8 x   8 x (512 + 512) =   8 x   8 x 1024
            nn.Sequential(nn.ConvTranspose2d( 1024, 512, 4, stride=2, padding=1 ), nn.BatchNorm2d( 512 ),               nn.ReLU()), #  16 x  16 x (512 + 512) =  16 x  16 x 1024
            nn.Sequential(nn.ConvTranspose2d( 1024, 256, 4, stride=2, padding=1 ), nn.BatchNorm2d( 256 ),               nn.ReLU()), #  32 x  32 x (256 + 256) =  32 x  32 x  512
            nn.Sequential(nn.ConvTranspose2d(  512, 128, 4, stride=2, padding=1 ), nn.BatchNorm2d( 128 ),               nn.ReLU()), #  64 x  64 x (128 + 128) =  64 x  64 x  256
            nn.Sequential(nn.ConvTranspose2d(  256,  64, 4, stride=2, padding=1 ), nn.BatchNorm2d(  64 ),               nn.ReLU()), # 128 x 128 x ( 64 +  64) = 128 x 128 x  128
            nn.Sequential(nn.ConvTranspose2d(  128,   3, 4, stride=2, padding=1 ), nn.BatchNorm2d(   3 ),               nn.ReLU()), #                           256 x 256 x    3
        ]

        self.upsampler1 = self.upsamplers[0]
        self.upsampler2 = self.upsamplers[1]
        self.upsampler3 = self.upsamplers[2]
        self.upsampler4 = self.upsamplers[3]
        self.upsampler5 = self.upsamplers[4]
        self.upsampler6 = self.upsamplers[5]
        self.upsampler7 = self.upsamplers[6]
        self.upsampler8 = self.upsamplers[7]

    def forward(self, x):
        skips = []
        t = x
        for layer in self.downsamplers:
            t = layer(t)
            skips.append(t)

        t = self.upsamplers[0](skips.pop())

        for layer in self.upsamplers[1:]:
            t = torch.cat((t, skips.pop()), dim=1)
            t = layer(t)

        return t

# Вход модели -- конкатенация входного изображения
# и результата генератора как отдельных каналов
class MyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(                                                                                                      # 256 x 256 x   6
            nn.Conv2d(   6,  64, 4, stride=2, padding=1, padding_mode='replicate', bias=False ),                        nn.LeakyReLU(), # 128 x 128 x  64
            nn.Conv2d(  64, 128, 4, stride=2, padding=1, padding_mode='replicate', bias=False ), nn.BatchNorm2d( 128 ), nn.LeakyReLU(), #  64 x  64 x 128
            nn.Conv2d( 128, 256, 4, stride=2, padding=1, padding_mode='replicate', bias=False ), nn.BatchNorm2d( 256 ), nn.LeakyReLU(), #  32 x  32 x 256
            nn.ZeroPad2d(1),                                                                                                            #  34 x  34 x 256
            nn.Conv2d( 256, 512, 4, bias=False ), nn.BatchNorm2d(512), nn.LeakyReLU(),                                                  #  31 x  31 x 512
            nn.ZeroPad2d(1),                                                                                                            #  33 x  33 x 512
            nn.Conv2d( 512,   1, 4, bias=False ),                                                                                       #  30 x  30 x   1
        )

    def forward(self, x):
        return self.main(x)

from torch.nn import functional as F

LAMBDA = 100

def generator_loss(disc_out, gen_out, target):
    gan_loss = F.binary_cross_entropy(torch.sigmoid(disc_out), torch.ones_like(disc_out))
    l1_loss = torch.mean(torch.abs(target - gen_out))
    total_loss = gan_loss + LAMBDA * l1_loss
    return total_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_out, disc_gen_out):
    real_loss = F.binary_cross_entropy(torch.sigmoid(disc_real_out), torch.ones_like(disc_real_out))
    gen_loss = F.binary_cross_entropy(torch.sigmoid(disc_gen_out), torch.zeros_like(disc_gen_out))
    return real_loss + gen_loss
