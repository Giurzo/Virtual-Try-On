import torch
import torch.nn as nn

from model_block import DownBlock, UpBlock, ConvBlock

class Generator(nn.Module):
    def __init__(self, in_channels=1, levels=6):
        super().__init__()

        self.initial = ConvBlock(in_channels,64)

        self.down = nn.ModuleList()
        for i in range(1,levels+1):
            self.down += [ DownBlock(64*i, 64*(i+1)) ]
        
        self.up = nn.ModuleList()
        for i in range(levels,0,-1):
            self.up += [ UpBlock(2 * 64*(i+1), 64*i) ]

        self.final = ConvBlock(2 * 64, 3,False,False)
            
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
    model = Generator(3, 4)
    from torchsummary import summary
    summary(model.cuda(), (3, 256, 192), 4, "cuda")


if __name__ == "__main__":
    test()