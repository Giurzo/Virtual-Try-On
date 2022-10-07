import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, relu=True):
        super(ConvBlock, self).__init__()
        layers = []
        layers += [ nn.utils.weight_norm( nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False, padding_mode="reflect") ) ]
        #layers += [ nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False, padding_mode="reflect") ]
        layers += [ nn.BatchNorm2d(out_channels) ] if norm else []
        layers += [ nn.LeakyReLU(0.2) ] if relu else []
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, relu=True):
        super(DownBlock, self).__init__()
        layers = []
        layers += [ nn.utils.weight_norm( nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") ) ]
        #layers += [ nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") ]
        layers += [ nn.BatchNorm2d(out_channels) ] if norm else []
        layers += [ nn.LeakyReLU(0.2) ] if relu else []
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, relu=True):
        super(UpBlock, self).__init__()
        layers = []
        layers += [ nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False) ]
        #layers += [ nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=(not norm)) ]
        layers += [ nn.BatchNorm2d(out_channels) ] if norm else []
        layers += [ nn.LeakyReLU(0.2) ] if relu else []
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def test():
    conv = ConvBlock(64, 512, True, True)
    down = DownBlock(64, 512, True, True)
    up = UpBlock(64, 512, True, True)
    from torchsummary import summary
    summary(conv.cuda(), (64, 256, 256))
    #summary(down.cuda(), (64, 256, 256))
    #summary(up.cuda(), (64, 256, 256))

if __name__ == "__main__":
    test()