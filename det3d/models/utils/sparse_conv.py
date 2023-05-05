import numpy as np

import spconv
import spconv.pytorch

from torch import nn
from torch.nn import functional as F
from spconv.core import ConvAlgo

from hydra.utils import instantiate



def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

class SparseConvBlock(spconv.pytorch.SparseModule):
    '''
    Sparse Conv Block
    SparseConv2d for stride > 1 and subMconv2d for stride==1
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, indice_key=None,norm_cfg=None, act_cfg=None, use_subm=True, bias=False):
        super(SparseConvBlock, self).__init__()
        if stride == 1 and use_subm:
            self.conv = spconv.pytorch.SubMConv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2,
                                                    stride=1, bias=bias, algo=ConvAlgo.Native, indice_key=indice_key)

        else:
            self.conv = spconv.pytorch.SparseConv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2,
                                                    stride=stride, bias=bias, algo=ConvAlgo.Native, indice_key=indice_key)

        if norm_cfg is None:
            norm_cfg = {'_target_': 'torch.nn.BatchNorm1d', 'eps': 1e-3, 'momentum': 0.01}

        self.norm = instantiate(norm_cfg, out_channels)
        
        if act_cfg is None:
            act_cfg = {'_target_': 'torch.nn.ReLU'}
        
        self.act = instantiate(act_cfg)
    
    def forward(self, x):
        out = self.conv(x)
        out = replace_feature(out, self.norm(out.features))
        out = replace_feature(out, self.act(out.features))

        return out
    

class SparseInverseConvBlock(spconv.pytorch.SparseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        indice_key=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        
        self.conv = spconv.pytorch.SparseInverseConv2d(in_channels, out_channels, kernel_size=kernel_size, indice_key=indice_key, algo=ConvAlgo.Native)

        if norm_cfg is None:
            norm_cfg = {'_target_': 'torch.nn.BatchNorm1d', 'eps': 1e-3, 'momentum': 0.01}
        self.norm = instantiate(norm_cfg, out_channels)
        
        if act_cfg is None:
            act_cfg = {'_target_': 'torch.nn.ReLU'}
        self.act = instantiate(act_cfg)
        
    def forward(self, x):
        out = self.conv(x)
        out = replace_feature(out, self.norm(out.features))
        out = replace_feature(out, self.act(out.features))

        return out
    

class SparseBasicBlock(spconv.pytorch.SparseModule):
    '''
    Sparse Conv Block
    ''' 
    def __init__(self, channels, kernel_size, indice_key=None, norm_cfg=None, act_cfg=None):
        super(SparseBasicBlock, self).__init__()
        self.block1 = SparseConvBlock(channels, channels, kernel_size, 1, indice_key, norm_cfg, act_cfg)
        self.conv2 = spconv.pytorch.SubMConv2d(channels, channels, kernel_size, padding=kernel_size//2,
                                                    stride=1, bias=False, algo=ConvAlgo.Native, indice_key=indice_key)
        if norm_cfg is None:
            norm_cfg = {'_target_': 'torch.nn.BatchNorm1d', 'eps': 1e-3, 'momentum': 0.01}
        self.norm2 = instantiate(norm_cfg, channels)
        if act_cfg is None:
            act_cfg = {'_target_': 'torch.nn.ReLU'}
        self.act2 = instantiate(act_cfg)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.act2(out.features))

        return out


class SparseConv3dBlock(spconv.pytorch.SparseModule):

    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_cfg=None, act_cfg=None, use_subm=True):
        super(SparseConv3dBlock, self).__init__()
        if stride == 1 and use_subm:
            self.conv = spconv.pytorch.SubMConv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2,
                                                    stride=1, bias=False, algo=ConvAlgo.Native)

        else:
            self.conv = spconv.pytorch.SparseConv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2,
                                                    stride=stride, bias=False, algo=ConvAlgo.Native)

        if norm_cfg is None:
            norm_cfg = {'_target_': 'torch.nn.BatchNorm1d', 'eps': 1e-3, 'momentum': 0.01}

        self.norm = instantiate(norm_cfg, out_channels)
        
        if act_cfg is None:
            act_cfg = {'_target_': 'torch.nn.ReLU'}
        
        self.act = instantiate(act_cfg)
    
    def forward(self, x):
        out = self.conv(x)
        out = replace_feature(out, self.norm(out.features))
        out = replace_feature(out, self.act(out.features))

        return out


class SparseBasicBlock3d(spconv.pytorch.SparseModule):
    '''
    Sparse Conv Block
    ''' 
    def __init__(self, channels, kernel_size, norm_cfg=None, act_cfg=None):
        super(SparseBasicBlock3d, self).__init__()
        self.block1 = SparseConv3dBlock(channels, channels, kernel_size, 1, norm_cfg, act_cfg)
        self.conv2 = spconv.pytorch.SubMConv3d(channels, channels, kernel_size, padding=kernel_size//2,
                                                    stride=1, bias=False, algo=ConvAlgo.Native)
        if norm_cfg is None:
            norm_cfg = {'_target_': 'torch.nn.BatchNorm1d', 'eps': 1e-3, 'momentum': 0.01}

        self.norm2 = instantiate(norm_cfg, channels)
        
        if act_cfg is None:
            act_cfg = {'_target_': 'torch.nn.ReLU'}
        
        self.act2 = instantiate(act_cfg)
        
    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.act2(out.features))

        return out
    

