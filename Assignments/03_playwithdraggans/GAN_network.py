import torch
import torch.nn as nn

# 生成器（U-Net）
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1, padding_mode='reflect'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 更多编码器层...
            nn.Conv2d(128, 256, 4, 2, 1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        # 定义底部
        self.bottom = nn.Sequential(
            nn.Conv2d(256, 256, 4, 2, 1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 4, 2, 1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 4, 2, 1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        # 定义解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 更多解码器层...
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 定义输出层
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottom(x)
        x = self.decoder(x)
        x = self.output(x)
        return x

# 判别器（PatchGAN）
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, x):
        # 生成器前向传播
        generated_image = self.generator(x)
        # 判别器前向传播
        combined_output = torch.cat((generated_image, x), dim=1)
        discriminator_output = self.discriminator(combined_output)
        return discriminator_output, generated_image