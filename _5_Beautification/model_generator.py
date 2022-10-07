import torch
import torch.nn as nn

import os, sys
sys.path.append(os.getcwd())
from __Utils.model_block import DownBlock, UpBlock, ConvBlock

class UNet(nn.Module):
    def __init__(self, in_channels=3, levels=6, out_channels=16):
        super().__init__()

        self.initial = ConvBlock(in_channels,64)

        self.down = nn.ModuleList()
        for i in range(1,levels+1):
            self.down += [ DownBlock(64*i, 64*(i+1)) ]
        
        self.up = nn.ModuleList()
        for i in range(levels,0,-1):
            self.up += [ UpBlock(2 * 64*(i+1), 64*i) ]

        self.final = ConvBlock(2 * 64, out_channels, False, False)
            
        self.tan = nn.Tanh()
        self.sig = nn.Sigmoid()
        

    def forward(self, x):
        x = self.initial(x)
        residual = [x]
        for layer in self.down:
            x = layer(x)
            residual += [x]
        for layer, res in zip(self.up,residual[:0:-1]):
            x = layer(torch.concat([x,res], 1))
        
        x = self.final(torch.concat([x,residual[0]], 1))

        return self.sig(x)


def test():
    model = UNet(3, 6)
    from torchsummary import summary
    summary(model.cuda(), (3, 256, 192), 1, "cuda")


if __name__ == "__main__":
    test()