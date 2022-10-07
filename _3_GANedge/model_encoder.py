import torch
import torch.nn as nn

from model_block import DownBlock, UpBlock, ConvBlock

class ColorEncoder(nn.Module):
    def __init__(self, in_channels=3, levels=6):
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


def test():
    model = ColorEncoder(3, 8)
    from torchsummary import summary
    summary(model.cuda(), (3, 256), 4, "cuda")


if __name__ == "__main__":
    test()