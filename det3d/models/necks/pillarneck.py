import torch
import torch.nn as nn
import numpy as np
from det3d.models.utils.conv import BasicBlock, ConvBlock
import torch.utils.checkpoint as cp
from trainer.utils import force_fp32

class PillarNeck(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_cfg=None,
        norm_cfg=None,
        deconv_cfg=None):

        super(PillarNeck, self).__init__()
        self._num_filters = in_channels

        if deconv_cfg is None:
            deconv_cfg = {'_target_': 'torch.nn.ConvTranspose2d', 'padding': 0, 'bias': False}
        if conv_cfg is None:
            conv_cfg = {'_target_': 'torch.nn.Conv2d', 'padding': 0, 'bias': False}
        
        self.conv1 = BasicBlock(in_channels, in_channels)
        self.downsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=conv_cfg)
        self.conv2 = BasicBlock(in_channels, in_channels)
        self.upsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=deconv_cfg)

        #self.fuse = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1)
        
    def _forward(self, x):
        x_high = self.conv1(x)
        x_low = self.downsample(x)
        x_low = self.conv2(x_low)
        x_low = self.upsample(x_low)
        
        out = x_high + x_low
        #out = self.fuse(out)

        return out

    def forward(self, x):
        if x.requires_grad:
            out = cp.checkpoint(self._forward, x)
        else:
            out = self._forward(x)

        return out
