import math
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision.ops import DeformConv2d
from functools import partial

from einops import rearrange, reduce
from timm.layers.weight_init import trunc_normal_
from timm.layers.activations import *
from timm.layers import DropPath
# from mmcv.ops import DeformConv2dPack

# ========== For Common ==========
class LayerNorm2d(nn.Module):
	
	def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
		super().__init__()
		self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
	
	def forward(self, x):
		x = rearrange(x, 'b c h w -> b h w c').contiguous()
		x = self.norm(x)
		x = rearrange(x, 'b h w c -> b c h w').contiguous()
		return x

def get_conv(conv_layer='conv_2d'):
	conv_dict = {
		'conv_2d': nn.Conv2d,
		'dcn2_2d': DCN2,
		# 'dcn2_2d_mmcv': DeformConv2dPack,
		'conv_2ds': Conv2ds,
	}
	return conv_dict[conv_layer]


def get_norm(norm_layer='in_1d'):
	eps = 1e-6
	norm_dict = {
		'none': nn.Identity,
		'in_1d': partial(nn.InstanceNorm1d, eps=eps),
		'in_2d': partial(nn.InstanceNorm2d, eps=eps),
		'in_3d': partial(nn.InstanceNorm3d, eps=eps),
		'bn_1d': partial(nn.BatchNorm1d, eps=eps),
		'bn_2d': partial(nn.BatchNorm2d, eps=eps),
		# 'bn_2d': partial(nn.SyncBatchNorm, eps=eps),
		'bn_3d': partial(nn.BatchNorm3d, eps=eps),
		'gn': partial(nn.GroupNorm, eps=eps),
		'ln_1d': partial(nn.LayerNorm, eps=eps),
		'ln_2d': partial(LayerNorm2d, eps=eps),
		'bn_2ds': partial(BatchNorm2ds, eps=eps),
	}
	return norm_dict[norm_layer]


def get_act(act_layer='relu'):
	act_dict = {
		'none': nn.Identity,
		'sigmoid': Sigmoid,
		'swish': Swish,
		'mish': Mish,
		'hsigmoid': HardSigmoid,
		'hswish': HardSwish,
		'hmish': HardMish,
		'tanh': Tanh,
		'relu': nn.ReLU,
		'relu6': nn.ReLU6,
		'prelu': PReLU,
		'gelu': GELU,
		'silu': nn.SiLU
	}
	return act_dict[act_layer]


class LayerScale(nn.Module):
	def __init__(self, dim, init_values=1e-5, inplace=True):
		super().__init__()
		self.inplace = inplace
		self.gamma = nn.Parameter(init_values * torch.ones(1, 1, dim))
	
	def forward(self, x):
		return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerScale2D(nn.Module):
	def __init__(self, dim, init_values=1e-5, inplace=True):
		super().__init__()
		self.inplace = inplace
		self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))
	
	def forward(self, x):
		return x.mul_(self.gamma) if self.inplace else x * self.gamma
	

class DCN2(nn.Module):
	# ref: https://github.com/WenmuZhou/DBNet.pytorch/blob/678b2ae55e018c6c16d5ac182558517a154a91ed/models/backbone/resnet.py
	def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False, deform_groups=4):
		super().__init__()
		offset_channels = kernel_size * kernel_size * 2
		self.conv_offset = nn.Conv2d(dim_in, deform_groups * offset_channels, kernel_size=3, stride=stride, padding=1)
		self.conv = DeformConv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

	def forward(self, x):
		offset = self.conv_offset(x)
		x = self.conv(x, offset)
		return x


class Conv2ds(nn.Conv2d):

	def __init__(self, dim_in, dim_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
				 padding_mode='zeros', device=None, dtype=None):
		super().__init__(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
		self.in_channels_i = dim_in
		self.out_channels_i = dim_out
		self.groups_i = groups

	def forward(self, x, dim_in=None, dim_out=None):
		self.groups = dim_in if self.groups_i != 1 else self.groups_i
		in_channels = dim_in if dim_in else self.in_channels_i
		out_channels = dim_out if dim_out else self.out_channels_i
		weight = self.weight[:out_channels, :in_channels, :, :]
		bias = self.bias[:out_channels] if self.bias is not None else self.bias
		return self._conv_forward(x, weight, bias)

class BatchNorm2ds(nn.BatchNorm2d):

	def __init__(self, dim_in, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
		super().__init__(dim_in, eps, momentum, affine, track_running_stats, device, dtype)
		self.num_features = dim_in

	def forward(self, x, dim_in=None):
		self._check_input_dim(x)
		if self.momentum is None:
			exponential_average_factor = 0.0
		else:
			exponential_average_factor = self.momentum
		if self.training and self.track_running_stats:
			if self.num_batches_tracked is not None:  # type: ignore[has-type]
				self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
				if self.momentum is None:  # use cumulative moving average
					exponential_average_factor = 1.0 / float(self.num_batches_tracked)
				else:  # use exponential moving average
					exponential_average_factor = self.momentum
		if self.training:
			bn_training = True
		else:
			bn_training = (self.running_mean is None) and (self.running_var is None)
		running_mean = self.running_mean[:dim_in]
		running_var = self.running_var[:dim_in]
		weight = self.weight[:dim_in]
		bias = self.bias[:dim_in]
		return F.batch_norm(x,
			running_mean if not self.training or self.track_running_stats else None,
			running_var if not self.training or self.track_running_stats else None,
			weight, bias, bn_training, exponential_average_factor, self.eps,
		)


class ConvNormAct(nn.Module):
	
	def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False, padding_mode='zeros', skip=False, conv_layer='conv_2d', norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
		super(ConvNormAct, self).__init__()
		self.conv_layer = conv_layer
		self.norm_layer = norm_layer
		self.has_skip = skip and dim_in == dim_out
		padding = math.ceil(((kernel_size - 1) * dilation + 1 - stride) / 2)
		if conv_layer in ['conv_2d', 'conv_2ds']:
			self.conv = get_conv(conv_layer)(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_mode)
		elif conv_layer in ['dcn2_2d', 'dcn2_2d_mmcv']:
			self.conv = get_conv(conv_layer)(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, deform_groups=4, bias=bias)
		self.norm = get_norm(norm_layer)(dim_out)
		self.act = get_act(act_layer)(inplace=inplace)
		self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()
	
	def forward(self, x, dim_in=None, dim_out=None):
		shortcut = x
		x = self.conv(x, dim_in=dim_in, dim_out=dim_out) if self.conv_layer in ['conv_2ds'] else self.conv(x)
		x = self.norm(x, dim_in=dim_out) if self.norm_layer in ['bn_2ds'] else self.norm(x)
		x = self.act(x)
		if self.has_skip:
			x = self.drop_path(x) + shortcut
		return x


def gen_cfg(opss=['d2.0', 'd3.0', 's1.0d3.0', 's1.0d3.0'], depths=[3, 3, 9, 3]):
	cfg = []
	for ops, depth in zip(opss, depths):
		ops = re.findall(r'[a-zA-Z]\d\.\d*', ops)
		ops = {op[0]: float(op[1:]) for op in ops}
		for i in range(depth):
			if i == 0:
				cfg += ['d{:.3f}'.format(sum(list(ops.values())) * 2)]
			else:
				cfg_l = ''
				for k, v in ops.items():
					cfg_l += '{}{:.3f}'.format(k, v)
				cfg += [cfg_l]
	return cfg


if __name__ == '__main__':
	x = torch.rand(2, 3, 224).cuda()
