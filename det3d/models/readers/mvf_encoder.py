"""
authored by Beijing-jinyu
"""
import torch
from torch import nn
from torch.nn import functional as F

import spconv
import spconv.pytorch

import numpy as np
import torch_scatter
from functools import reduce

from det3d.models.utils.sparse_conv import SparseConvBlock, SparseBasicBlock
from det3d.models.readers.pillar_encoder import PFNLayer


class PointNet(nn.Module):
    """
    Linear Process for point feature
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, points):
        torch.backends.cudnn.enabled = False
        x = self.linear(points)
        x = self.norm(x)
        x = F.relu(x)
        torch.backends.cudnn.enabled = True

        return x


class PillarVoxelNet(nn.Module):
    def __init__(self, voxel_size, pc_range):
        super().__init__()
        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)

    def forward(self, points):
        device = points.device
        dtype = points.dtype

        grid_size = (self.pc_range[3:] - self.pc_range[:3]
                     )/self.voxel_size  # x,  y, z
        grid_size = np.round(grid_size, 0, grid_size).astype(np.int64)

        voxel_size = torch.from_numpy(
            self.voxel_size).type_as(points).to(device)
        pc_range = torch.from_numpy(self.pc_range).type_as(points).to(device)

        points_coords = (
            points[:, 1:4] - pc_range[:3].view(-1, 3)) / voxel_size.view(-1, 3)   # x, y, z
        points_coords[:, 0] = torch.clamp(
            points_coords[:, 0], 0, grid_size[0] - 1)
        points_coords[:, 1] = torch.clamp(
            points_coords[:, 1], 0, grid_size[1] - 1)
        points_coords[:, 2] = torch.clamp(
            points_coords[:, 2], 0, grid_size[2] - 1)

        points_coords = points_coords.long()
        batch_idx = points[:, 0:1].long()

        points_index = torch.cat((batch_idx, points_coords[:, :2]), dim=1)
        unq, unq_inv = torch.unique(points_index, return_inverse=True, dim=0)
        unq = unq.int()        # breakpoint()

        points_mean_scatter = torch_scatter.scatter_mean(
            points[:, 1:4], unq_inv, dim=0)

        f_cluster = points[:, 1:4] - points_mean_scatter[unq_inv]

        # Find distance of x, y, and z from pillar center
        f_center = points[:, 1:3] - (points_coords[:, :2].to(dtype) * voxel_size[:2].unsqueeze(0) +
                                     voxel_size[:2].unsqueeze(0) / 2 + pc_range[:2].unsqueeze(0))

        # Combine together feature decorations
        features = torch.cat([points[:, 1:], f_cluster, f_center], dim=-1)

        return features, unq[:, [0, 2, 1]], unq_inv, grid_size[[1, 0]]


class CylinderNet(nn.Module):
    def __init__(self, voxel_size, pc_range):
        super().__init__()
        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)

    def forward(self, points):
        device = points.device
        dtype = points.dtype
        points_x = points[:, 1:2]
        points_y = points[:, 2:3]
        points_z = points[:, 3:4]
        points_phi = torch.atan2(points_y, points_x) / np.pi * 180
        points_rho = torch.sqrt(points_x ** 2 + points_y ** 2)
        points_cylinder = torch.cat(
            (points[:, 0:1], points_phi, points_z, points_rho, points[:, 4:]), dim=-1)

        grid_size = (self.pc_range[3:] - self.pc_range[:3]
                     )/self.voxel_size  # phi, z, rho
        grid_size = np.round(grid_size, 0, grid_size).astype(np.int64)

        voxel_size = torch.from_numpy(
            self.voxel_size).type_as(points).to(device)
        pc_range = torch.from_numpy(self.pc_range).type_as(points).to(device)

        points_coords = (
            points_cylinder[:, 1:4] - pc_range[:3].view(-1, 3)) / voxel_size.view(-1, 3)
        points_coords[:, 0] = torch.clamp(
            points_coords[:, 0], 0, grid_size[0] - 1)
        points_coords[:, 1] = torch.clamp(
            points_coords[:, 1], 0, grid_size[1] - 1)
        points_coords[:, 2] = torch.clamp(
            points_coords[:, 2], 0, grid_size[2] - 1)
        points_coords = points_coords.long()
        batch_idx = points_cylinder[:, 0:1].long()

        points_index = torch.cat((batch_idx, points_coords[:, :2]), dim=1)
        unq, unq_inv = torch.unique(points_index, return_inverse=True, dim=0)
        unq = unq.int()

        points_mean_scatter = torch_scatter.scatter_mean(
            points_cylinder[:, 1:4], unq_inv, dim=0)
        f_cluster = points_cylinder[:, 1:4] - points_mean_scatter[unq_inv]

        # Find distance of x, y, and z from pillar center
        f_center = points_cylinder[:, 1:3] - (points_coords[:, :2].to(dtype) * voxel_size[:2].unsqueeze(0) +
                                              voxel_size[:2].unsqueeze(0) / 2 + pc_range[:2].unsqueeze(0))

        # Combine together feature decorations
        features = torch.cat(
            [points_cylinder[:, 1:], f_cluster, f_center], dim=-1)

        return features, unq[:, [0, 2, 1]], unq_inv, grid_size[[1, 0]]


class SingleView(nn.Module):
    """
    authoured by Beijing-jinyu
    convolution for single view
    """

    def __init__(self, in_channels, num_filters, layer_nums, ds_layer_strides, ds_num_filters, kernel_size, mode, voxel_size, pc_range, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.mode = mode
        self.voxel_size = np.array(voxel_size[:2])
        self.bias = np.array(pc_range[:2])
        num_filters = [in_channels] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        in_filters = [num_filters[-1], *ds_num_filters[:-1]]
        self.pfn_layers = nn.ModuleList(pfn_layers)
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = self._make_layer(
                in_filters[i],
                ds_num_filters[i],
                kernel_size[i],
                ds_layer_strides[i],
                layer_num)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.ds_rate = np.prod(np.array(ds_layer_strides))

    def _make_layer(self, inplanes, planes, kernel_size, stride, num_blocks):

        layers = []
        layers.append(SparseConvBlock(inplanes, planes,
                      kernel_size=kernel_size, stride=stride, use_subm=False))

        for j in range(num_blocks):
            layers.append(SparseBasicBlock(planes, kernel_size=kernel_size))

        return spconv.pytorch.SparseSequential(*layers)

    def forward(self, features, unq, unq_inv, grid_size):
        feature_pos = features[:,
                               0:2] if self.mode == 'pillar' else features[:, 10:12]
        device = feature_pos.device
        voxel_size = torch.from_numpy(
            self.voxel_size).type_as(feature_pos).to(device)
        bias = torch.from_numpy(self.bias).type_as(feature_pos).to(device)
        feature_pos = (feature_pos - bias) / voxel_size

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)  # num_points, dim_feat
        features_voxel = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        batch_size = len(torch.unique(unq[:, 0]))
        x = spconv.pytorch.SparseConvTensor(
            features_voxel, unq, grid_size, batch_size)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = x.dense()
        feature_pos = torch.cat(
            (unq[unq_inv][:, 0:1], feature_pos / self.ds_rate), dim=-1)

        return self.bilinear_interpolate(x, feature_pos)

    def bilinear_interpolate(self, image, coords):
        """
        image: (B, C, H, W)
        coords: (N, 3): (B, y, x)
        """
        x = coords[:, 1]
        x0 = torch.floor(x).long()
        x1 = x0 + 1

        y = coords[:, 2]
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        B = coords[:, 0].long()

        x0 = torch.clamp(x0, 0, image.shape[3] - 1)
        x1 = torch.clamp(x1, 0, image.shape[3] - 1)
        y0 = torch.clamp(y0, 0, image.shape[2] - 1)
        y1 = torch.clamp(y1, 0, image.shape[2] - 1)

        Ia = image[B, :, y0, x0]
        Ib = image[B, :, y1, x0]
        Ic = image[B, :, y0, x1]
        Id = image[B, :, y1, x1]

        wa = ((x1.type(torch.float32)-x) *
              (y1.type(torch.float32)-y)).unsqueeze(-1)
        wb = ((x1.type(torch.float32)-x) *
              (y-y0.type(torch.float32))).unsqueeze(-1)
        wc = ((x-x0.type(torch.float32)) *
              (y1.type(torch.float32)-y)).unsqueeze(-1)
        wd = ((x-x0.type(torch.float32)) *
              (y-y0.type(torch.float32))).unsqueeze(-1)

        features = Ia * wa + Ib * wb + Ic * wc + Id * wd

        return features


class MVFFeatureNet(nn.Module):
    """
    authoured by Beijing-jinyu
    """

    def __init__(self,
                 in_channels,
                 voxel_size,
                 pc_range,
                 cylinder_size,
                 cylinder_range,
                 num_filters,
                 layer_nums,
                 ds_layer_strides,
                 ds_num_filters,
                 kernel_size,
                 out_channels
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.cylinder_range = cylinder_range
        self.cylinder_size = cylinder_size

        self.voxelization = PillarVoxelNet(voxel_size, pc_range)
        self.cylinderlization = CylinderNet(cylinder_size, cylinder_range)

        self.pillarview = SingleView((in_channels + 5) * 2, num_filters, layer_nums, ds_layer_strides,
                                     ds_num_filters, kernel_size, 'pillar', self.voxel_size, self.pc_range)
        self.cylinderview = SingleView((in_channels + 5) * 2, num_filters, layer_nums, ds_layer_strides,
                                       ds_num_filters, kernel_size, 'cylinder', self.cylinder_size, self.cylinder_range)
        self.ds_rate = np.prod(np.array(ds_layer_strides))

        self.pointnet1 = PointNet((in_channels + 5) * 2, ds_num_filters[-1])
        self.pointnet2 = PointNet(ds_num_filters[-1] * 3, out_channels)

    def forward(self, points):
        dtype = points.dtype
        pc_range = torch.tensor(self.pc_range, dtype=dtype)
        mask = reduce(torch.logical_and, (points[:, 1] >= pc_range[0],
                                          points[:, 1] < pc_range[3],
                                          points[:, 2] >= pc_range[1],
                                          points[:, 2] < pc_range[4],
                                          points[:, 3] >= pc_range[2],
                                          points[:, 3] < pc_range[5]))
        points = points[mask]

        pillar_feature, pillar_coords, pillar_inv, pillar_size = self.voxelization(
            points)
        cylinder_feature, cylinder_coords, cylinder_inv, cylinder_size = self.cylinderlization(
            points)
        points_feature = torch.cat((pillar_feature, cylinder_feature), dim=-1)

        pillar_view = self.pillarview(
            points_feature, pillar_coords, pillar_inv, pillar_size)
        cylinder_view = self.cylinderview(
            points_feature, cylinder_coords, cylinder_inv, cylinder_size)

        points_feature = self.pointnet1(points_feature)
        points_feature = torch.cat(
            (points_feature, pillar_view, cylinder_view), dim=-1)
        pillar_feature = self.pointnet2(points_feature)
        pillar_feature = torch_scatter.scatter_max(
            pillar_feature, pillar_inv, dim=0)[0]
        batch_size = len(torch.unique(pillar_coords[:, 0]))
        pillar_coords[:, 1:] = pillar_coords[:, 1:] // self.ds_rate
        pillar_size = pillar_size // self.ds_rate
        x = spconv.pytorch.SparseConvTensor(
            pillar_feature, pillar_coords, pillar_size, batch_size)
        return x.dense()
