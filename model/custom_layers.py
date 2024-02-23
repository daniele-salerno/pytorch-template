import torch.nn as nn

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super().__init__()
        self.depthwise_layer = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3,3), groups=in_channels)
        self.pointwise_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise_layer(x)
        x = self.pointwise_layer(x)
        return x
