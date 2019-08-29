from torch import nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.mask_type = mask_type
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kernel_height, kernel_width = self.weight.size()
        self.mask.fill_(0)
        half_h, half_w = kernel_height // 2, kernel_width // 2

        self.mask[:, :, :half_h, :] = 1.0
        self.mask[:, :, half_h, :half_w] = 1.0
        if self.mask_type == 'A':
            self.mask[:, :, half_h, half_w] = 0.0
        else:
            self.mask[:, :, half_h, half_w] = 1.0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class ResidualBlock(nn.Module):
    def __init__(self, h):
        super(ResidualBlock, self).__init__()
        self.h = h
        # todo: set padding to same
        self.network = []
        self.network.extend([
            nn.Conv2d(self.h, self.h // 2, (1, 1)),
            nn.BatchNorm2d(self.h // 2),
            nn.ReLU()
        ])
        self.network.extend([
            MaskedConv2d('B', self.h // 2, self.h // 2, (3, 3), padding=1),
            nn.BatchNorm2d(self.h // 2),
            nn.ReLU()
        ])

        self.network.extend([
            nn.Conv2d(self.h // 2, self.h, (1, 1)),
            nn.BatchNorm2d(self.h),
            nn.ReLU()
        ])

        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        skip = x
        x = self.network(x)
        return F.relu(x + skip)


class PixelCNN(nn.Module):
    def __init__(self, in_channels):
        super(PixelCNN, self).__init__()

        self.in_channels = in_channels

        self.network = [nn.BatchNorm2d(3)]

        # 7x7 Conv input, type A
        self.network.extend([
            MaskedConv2d('A', 3, self.in_channels, (7, 7), padding=3),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        ])

        self.network.extend(
            [ResidualBlock(self.in_channels) for _ in range(12)]
        )

        #         # 3x3 Conv input, type B
        #         self.network.extend([
        #             MaskedConv2d('B', self.in_channels, self.in_channels, (3, 3), padding=1),
        #             nn.BatchNorm2d(self.in_channels),
        #             nn.ReLU(),
        #         ])

        #         # 1x1 Conv input
        self.network.extend([
            nn.Conv2d(self.in_channels, self.in_channels, (1, 1)),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()])
        self.network.extend([nn.Conv2d(self.in_channels, 12, (1, 1))])

        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        x = self.network(x)
        x = x.reshape(x.size(0), 4, 3, 28, 28)
        return x
