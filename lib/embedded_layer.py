import torch.nn as nn


class embed_layer(nn.Module):
    def __init__(self, in_channels=2048, out_channels=128):
        super(embed_layer, self).__init__()
        self.conv_embedd = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
    
    def forward(self, x):
        x = self.conv_embedd(x)
        return x.flatten(2)
    