import torch
import torch.nn as nn

from model_block import DownBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256]):
        super().__init__()
        
        layers = []
        layers += [ DownBlock(in_channels, features[0], norm=False)]

        for feature in features:
            layers += [ DownBlock(feature, feature * 2) ]

        layers += [ DownBlock(features[-1] * 2, 1, norm=False, relu=False) ]

        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)


def test():
    model = Discriminator(in_channels=3)
    from torchsummary import summary
    summary(model.cuda(), (3, 256, 192))

if __name__ == "__main__":
    test()