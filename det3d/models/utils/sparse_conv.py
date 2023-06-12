from torch import nn
import spconv
import spconv.pytorch
from spconv.core import ConvAlgo


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

    def __init__(self, in_channels, out_channels, kernel_size, stride, use_subm=True, bias=False):
        super(SparseConvBlock, self).__init__()
        if stride == 1 and use_subm:
            self.conv = spconv.pytorch.SubMConv2d(in_channels, out_channels, kernel_size,
                                                  padding=kernel_size//2, stride=1, bias=bias,)
        else:
            self.conv = spconv.pytorch.SparseConv2d(in_channels, out_channels, kernel_size,
                                                    padding=kernel_size//2, stride=stride, bias=bias)

        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = replace_feature(out, self.norm(out.features))
        out = replace_feature(out, self.act(out.features))

        return out


class SparseBasicBlock(spconv.pytorch.SparseModule):
    '''
    Sparse Conv Block
    '''

    def __init__(self, channels, kernel_size):
        super(SparseBasicBlock, self).__init__()
        self.block1 = SparseConvBlock(channels, channels, kernel_size, 1)
        self.conv2 = spconv.pytorch.SubMConv2d(channels, channels, kernel_size, padding=kernel_size//2,
                                               stride=1, bias=False, algo=ConvAlgo.Native, )
        self.norm2 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.act2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.act2(out.features))

        return out


class SparseConv3dBlock(spconv.pytorch.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_subm=True):
        super(SparseConv3dBlock, self).__init__()
        if stride == 1 and use_subm:
            self.conv = spconv.pytorch.SubMConv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2,
                                                  stride=1, bias=False)
        else:
            self.conv = spconv.pytorch.SparseConv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2,
                                                    stride=stride, bias=False)

        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = replace_feature(out, self.norm(out.features))
        out = replace_feature(out, self.act(out.features))

        return out


class SparseBasicBlock3d(spconv.pytorch.SparseModule):
    def __init__(self, channels, kernel_size):
        super(SparseBasicBlock3d, self).__init__()
        self.block1 = SparseConv3dBlock(channels, channels, kernel_size, 1)
        self.conv2 = spconv.pytorch.SubMConv3d(channels, channels, kernel_size, padding=kernel_size//2,
                                               stride=1, bias=False)
        self.norm2 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.act2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.act2(out.features))

        return out
