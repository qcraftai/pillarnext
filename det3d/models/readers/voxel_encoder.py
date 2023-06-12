"""
created by Beijing-jinyu
"""
import torch
from torch import nn

import numpy as np
import torch_scatter
from functools import reduce


class DynamicVoxelEncoder(nn.Module):
    """
    dynamic version of VoxelFeatureExtractorV3
    """

    def __init__(self):
        super(DynamicVoxelEncoder, self).__init__()

    def forward(self, inputs, unq_inv):
        features = torch_scatter.scatter_mean(inputs, unq_inv, dim=0)

        return features


class VoxelNet(nn.Module):
    """
    dynamic voxelization for point clouds
    """

    def __init__(self, voxel_size, pc_range):
        super().__init__()
        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)

    def forward(self, points):
        """
        points: Tensor: (N, d), batch_id, x, y, z, ...
        """
        device = points.device

        # voxel range of x, y, z
        grid_size = (self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size
        grid_size = np.round(grid_size, 0, grid_size).astype(np.int64)

        voxel_size = torch.from_numpy(
            self.voxel_size).type_as(points).to(device)
        pc_range = torch.from_numpy(self.pc_range).type_as(points).to(device)

        points_coords = (
            points[:, 1:4] - pc_range[:3].view(-1, 3)) / voxel_size.view(-1, 3)  # x, y, z

        mask = reduce(torch.logical_and, (points_coords[:, 0] >= 0,
                                          points_coords[:, 0] < grid_size[0],
                                          points_coords[:, 1] >= 0,
                                          points_coords[:, 1] < grid_size[1],
                                          points_coords[:, 2] >= 0,
                                          points_coords[:, 2] < grid_size[2]))  # remove the points out of range

        points = points[mask]
        points_coords = points_coords[mask]

        points_coords = points_coords.long()
        batch_idx = points[:, 0:1].long()
        point_index = torch.cat((batch_idx, points_coords), dim=1)

        unq, unq_inv = torch.unique(point_index, return_inverse=True, dim=0)
        unq = unq.int()

        features = points[:, 1:]

        return features, unq[:, [0, 3, 2, 1]], unq_inv, grid_size[[2, 1, 0]]


class VoxelFeatureNet(nn.Module):
    def __init__(self, voxel_size, pc_range):
        super().__init__()

        self.voxelization = VoxelNet(voxel_size, pc_range)
        self.voxel_encoder = DynamicVoxelEncoder()

    def forward(self, points):
        features, coords, unq_inv, grid_size = self.voxelization(points)

        features = self.voxel_encoder(features, unq_inv)

        return features, coords, grid_size
