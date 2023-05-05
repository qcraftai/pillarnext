
import torch
import torch.nn as nn
import numpy as np
from hydra.utils import instantiate

class Conv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, conv_cfg=None, circular_padding=False):
        super(Conv, self).__init__()
        if conv_cfg is None:
            conv_cfg = {'_target_': 'torch.nn.Conv2d', 'bias': False}
        self.circular_padding = circular_padding
        self.kernel_size = kernel_size
        conv_cfg['kernel_size'] = kernel_size
        conv_cfg['stride'] = stride
        if 'padding' in conv_cfg or circular_padding:
            self.conv = instantiate(conv_cfg, inplanes, planes)
        else:
            self.conv = instantiate(conv_cfg, inplanes, planes, padding=kernel_size//2)
                        
    def forward(self, x):
        if self.circular_padding: # todo, currently only support conv2d single dim padding
            padding_size = self.kernel_size//2
            x = nn.ZeroPad2d(1)(x)
            x[:, :, padding_size:-padding_size, 0:padding_size] = x[:, :, padding_size:-padding_size, -2*padding_size:-padding_size]
            x[:, :, padding_size:-padding_size, -padding_size:] = x[:, :, padding_size:-padding_size, padding_size:2*padding_size]
        
        return self.conv(x)




class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, circular_padding=False, 
                        conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(ConvBlock, self).__init__()
        self.conv = Conv(inplanes, planes, kernel_size=kernel_size, stride=stride, conv_cfg=conv_cfg, circular_padding=circular_padding)
        if norm_cfg is None:
            norm_cfg = {'_target_': 'torch.nn.BatchNorm2d', 'eps': 1e-3, 'momentum': 0.01}
        self.norm = instantiate(norm_cfg, planes)
        
        if act_cfg is None:
            act_cfg = {'_target_': 'torch.nn.ReLU'}
        
        self.act = instantiate(act_cfg)
    

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out



class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride= 1, conv_cfg=None, norm_cfg=None, act_cfg=None, circular_padding = False):
        super(BasicBlock, self).__init__()
        self.block1 = ConvBlock(inplanes, planes, kernel_size=kernel_size, stride=stride, 
                                  circular_padding=circular_padding, conv_cfg=conv_cfg, norm_cfg = norm_cfg, act_cfg=act_cfg)
        self.block2 = ConvBlock(inplanes, planes, kernel_size=kernel_size, stride=stride, 
                                  circular_padding=circular_padding, conv_cfg=conv_cfg, norm_cfg = norm_cfg, act_cfg=act_cfg)
        
        if stride != 1 or inplanes != planes:
            downsample_conv = Conv(inplanes, planes, kernel_size,  stride, conv_cfg=conv_cfg,circular_padding=circular_padding)
            if norm_cfg is None:
                norm_cfg = {'_target_': 'torch.nn.BatchNorm2d', 'eps': 1e-3, 'momentum': 0.01}

            downsample_norm = instantiate(norm_cfg, planes)
            self.downsample = nn.Sequential(downsample_conv, downsample_norm)
        else:
            self.downsample = None

        if act_cfg is None:
            act_cfg = {'_target_': 'torch.nn.ReLU'}
        
        self.act = instantiate(act_cfg)
    
    def forward(self, x):
        identity = x

        out = self.block1(x)
        out = self.block2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out = out + identity
        out = self.act(out)

        return out


class SASA_Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=7, num_heads=8, inference=False):
        super(SASA_Layer, self).__init__()
        self.kernel_size = kernel_size     
        self.num_heads = num_heads
        self.dk = self.dv = in_channels
        self.dkh = self.dk // self.num_heads
        self.dvh = self.dv // self.num_heads

        assert self.dk % self.num_heads == 0, "dk should be divided by num_heads. (example: dk: 32, num_heads: 8)"
        assert self.dk % self.num_heads == 0, "dv should be divided by num_heads. (example: dv: 32, num_heads: 8)"  
        
        self.k_conv = nn.Conv2d(self.dk, self.dk, kernel_size=1)
        self.q_conv = nn.Conv2d(self.dk, self.dk, kernel_size=1)
        self.v_conv = nn.Conv2d(self.dv, self.dv, kernel_size=1)
        
        # Positional encodings
        self.rel_encoding_h = nn.Parameter(torch.randn(self.dk // 2, self.kernel_size, 1), requires_grad=True)
        self.rel_encoding_w = nn.Parameter(torch.randn(self.dk // 2, 1, self.kernel_size), requires_grad=True)
        
        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)
            
    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Compute k, q, v
        padded_x = F.pad(x, [(self.kernel_size-1)//2, (self.kernel_size-1)-((self.kernel_size-1)//2), (self.kernel_size-1)//2, (self.kernel_size-1)-((self.kernel_size-1)//2)])
        k = self.k_conv(padded_x)
        q = self.q_conv(x)
        v = self.v_conv(padded_x)
        
        # Unfold patches into [BS, num_heads*depth, horizontal_patches, vertical_patches, kernel_size, kernel_size]
        k = k.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        v = v.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)

        # Reshape into [BS, num_heads, horizontal_patches, vertical_patches, depth_per_head, kernel_size*kernel_size]
        k = k.reshape(batch_size, self.num_heads, height, width, self.dkh, -1)
        v = v.reshape(batch_size, self.num_heads, height, width, self.dvh, -1)
        
        # Reshape into [BS, num_heads, height, width, depth_per_head, 1]
        q = q.reshape(batch_size, self.num_heads, height, width, self.dkh, 1)

        qk = torch.matmul(q.transpose(4, 5), k)    
        qk = qk.reshape(batch_size, self.num_heads, height, width, self.kernel_size, self.kernel_size)
        
        # Add positional encoding
        qr_h = torch.einsum('bhxydz,cij->bhxyij', q, self.rel_encoding_h)
        qr_w = torch.einsum('bhxydz,cij->bhxyij', q, self.rel_encoding_w)
        qk += qr_h
        qk += qr_w
        
        qk = qk.reshape(batch_size, self.num_heads, height, width, 1, self.kernel_size*self.kernel_size)
        weights = F.softmax(qk, dim=-1)    
        
        if self.inference:
            self.weights = nn.Parameter(weights)
        
        attn_out = torch.matmul(weights, v.transpose(4, 5)) 
        attn_out = attn_out.reshape(batch_size, -1, height, width)
        return attn_out


class SASABlock(nn.Module):
    def __init__(self, planes, kernel_size=19, num_heads=8):
        super(SASABlock, self).__init__()
        norm_cfg = {'_target_': 'torch.nn.BatchNorm2d', 'eps': 1e-3, 'momentum': 0.01}
        act_cfg = {'_target_': 'torch.nn.ReLU'}
        self.block1 = ConvBlock(planes, planes, kernel_size=1, stride=1)
        self.attention = SASA_Layer(planes, kernel_size, num_heads)
        self.norm1 = instantiate(norm_cfg, planes)
        self.conv = Conv(planes, planes, kernel_size=1, stride=1)
        self.norm2 = instantiate(norm_cfg, planes)
        self.act = instantiate(act_cfg)
    def forward(self, x):
        out = self.block1(x)
        out = self.attention(out)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv(out)
        out = self.norm2(out)
        
        out = x + out
        out = self.act(out)

        return out


class AsppBlock(nn.Module):
    def __init__(self, in_channels):
        super(AsppBlock, self).__init__()
        self.block1 = ConvBlock(in_channels, in_channels, kernel_size=1, stride=1, conv_cfg={'_target_': 'torch.nn.Conv2d', 'bias': False, 'padding': 0})
        self.block6 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, conv_cfg={'_target_': 'torch.nn.Conv2d', 'bias': False, 'padding': 6, 'dilation': 6})
        self.block12 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, conv_cfg={'_target_': 'torch.nn.Conv2d', 'bias': False, 'padding': 12, 'dilation': 12})
        self.block18 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, conv_cfg={'_target_': 'torch.nn.Conv2d', 'bias': False, 'padding': 18, 'dilation': 18})
        self.conv2 = ConvBlock(in_channels * 5, in_channels, kernel_size=1, stride=1, conv_cfg={'_target_': 'torch.nn.Conv2d', 'bias': False, 'padding': 0})

    def forward(self, x):
        return self.conv2(torch.cat((x, self.block1(x), self.block6(x), self.block12(x), self.block18(x)), dim=1))
