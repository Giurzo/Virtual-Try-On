import torch
import torch.nn as nn
import torch.nn.functional as F

from model_block import DownBlock, UpBlock, ConvBlock

class ColorEncoder(nn.Module):
    def __init__(self, in_channels=3, levels=8):
        super().__init__()

        self.initial = ConvBlock(in_channels,4, dim=1)

        self.down = nn.ModuleList()
        for i in range(1,levels+1):
            self.down += [ DownBlock(4*i, 4*(i+1), dim=1) ]

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down:
            x = layer(x)

        return x

class ColorDecoder(nn.Module):
    def __init__(self, in_channels=3, levels=8):
        super().__init__()

        self.up = nn.ModuleList()
        for i in range(levels,0,-1):
            self.up += [ UpBlock(4*(i+1), 4*i, dim=1) ]

        self.final = ConvBlock(4,in_channels, dim=1)

    def forward(self, x):
        for layer in self.up:
            x = layer(x)
        x = self.final(x)

        return x


class ColorAutoencoder(nn.Module):
    def __init__(self, in_channels=3, levels=8):
        super().__init__()

        self.enc = ColorEncoder(in_channels, levels)
        self.bneck = nn.utils.weight_norm( nn.Linear(36,4) )
        self.debneck = nn.utils.weight_norm( nn.Linear(4,36) )
        self.dec = ColorDecoder(in_channels, levels)

    def forward(self, x, vec_space=False):
        x = self.enc(x)

        x = x.squeeze(2)
        x = self.bneck(x)

        if vec_space:
            return x

        #x = torch.softmax(x, dim=1)
        x = F.leaky_relu(x)
        
        x = self.debneck(x)
        x = x.unsqueeze(2)

        x = F.leaky_relu(x,)
        
        x = self.dec(x)
        return F.sigmoid(x)


def test():
    from torchsummary import summary

    model = ColorEncoder(3, 8)
    summary(model.cuda(), (3, 256), 4, "cuda")

    model = ColorDecoder(3, 8)
    summary(model.cuda(), (36, 1), 4, "cuda")

    model = ColorAutoencoder(3, 8)
    summary(model.cuda(), (3, 256), 4, "cuda")


if __name__ == "__main__":
    test()