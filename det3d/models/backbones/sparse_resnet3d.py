import numpy as np

import spconv
import spconv.pytorch
import torch
from torch import nn
from torch.nn import functional as F
from spconv.core import ConvAlgo
from det3d.models.utils.sparse_conv import SparseConv3dBlock, SparseBasicBlock3d
from spconv.pytorch import SparseSequential, SparseConv3d
from hydra.utils import instantiate

class SparseResNet3D(spconv.pytorch.SparseModule):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        num_input_features,
        kernel_size=[3,3,3,3],
        norm_cfg=None,
        act_cfg=None,
        out_channels=128
    ):
        super(SparseResNet3D, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features


        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
    

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
       

        for i, layer_num in enumerate(self._layer_nums):
            block = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                kernel_size[i],
                self._layer_strides[i], 
                layer_num)
            blocks.append(block)
           
        self.blocks = nn.ModuleList(blocks)
        self.mapping = SparseConv3dBlock(self._num_filters[-1], out_channels, kernel_size=1, stride=1, use_subm=True)
        self.extra_conv = SparseSequential(
            SparseConv3d(
                self._num_filters[-1], self._num_filters[-1], (3, 1, 1), (2, 1, 1), bias=False, algo=ConvAlgo.Native),
            instantiate({'_target_': 'torch.nn.BatchNorm1d', 'eps': 1e-3, 'momentum': 0.01}, self._num_filters[-1]),
            nn.ReLU(),
        )
        
    def _make_layer(self, inplanes, planes, kernel_size, stride, num_blocks):

        layers = []
        layers.append(SparseConv3dBlock(inplanes, planes, kernel_size=kernel_size, stride=stride, use_subm=False))

        for j in range(num_blocks):
            layers.append(SparseBasicBlock3d(planes, kernel_size=kernel_size))

        return spconv.pytorch.SparseSequential(*layers)
    
    def forward(self, pillar_features, coors, input_shape):
        batch_size = len(torch.unique(coors[:, 0]))
        x = spconv.pytorch.SparseConvTensor(pillar_features, coors, input_shape, batch_size)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = self.extra_conv(x)
        x = self.mapping(x)
        x =  x.dense()
        B, C, D, H, W = x.shape
        x = x.view(B, C * D, H, W)
        return x



