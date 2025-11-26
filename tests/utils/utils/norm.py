#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch_npu
from torch.nn.parameter import Parameter
import torch.distributed as dist

from mindiesd.utils.utils import is_npu_available


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x, shift, scale):
        if is_npu_available():
            return torch_npu.npu_layer_norm_eval(
                x, normalized_shape=[self.hidden_size], weight=scale, bias=shift, eps=self.eps)
        else:
            return F.layer_norm(x, normalized_shape=[self.hidden_size], weight=scale, bias=shift, eps=self.eps)


class GroupNorm3dAdapter(nn.Module):
    def __init__(self, group_norm: nn.GroupNorm):
        super().__init__()
        self.module = PatchGroupNorm3d(
            num_groups=group_norm.num_groups,
            num_channels=group_norm.num_channels,
            eps=group_norm.eps,
            affine=group_norm.affine
        )
        if group_norm.affine:
            self.module.weight = group_norm.weight
            self.module.bias = group_norm.bias

    def forward(self, x):
        return self.module(x)


class PatchGroupNorm3d(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        self.init_paramsters(num_groups, num_channels, eps, affine)
        if self.affine:
            self.init_weight_bias()
        else:
            self.init_register_parameter()

        self.reset_parameters()

    def init_paramsters(self, num_groups, num_channels, eps, affine):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

    def init_weight_bias(self):
        self.weight = Parameter(torch.empty(self.num_channels, self.factory_kwargs))
        self.bias = Parameter(torch.empty(self.num_channels, self.factory_kwargs))

    def init_register_parameter(self):
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        rank = dist.get_rank()
        width = torch.tensor(x.shape[-1], dtype=torch.int64, device=x.device) - 1
        dist.all_reduce(width)

        channels_per_group = x.shape[1] // self.num_groups
        nelements_rank = channels_per_group * x.shape[-3] * x.shape[-2] * (x.shape[-1] - 1)
        nelements = channels_per_group * x.shape[-3] * x.shape[-2] * width

        x = x.view(x.shape[0], self.num_groups, -1, *x.shape[2:])
        if rank % 2 == 0:
            group_sum = x[..., :-1].sum(dim=(2, 3, 4, 5), dtype=x.dtype, keepdim=True)
        else:
            group_sum = x[..., 1:].sum(dim=(2, 3, 4, 5), dtype=x.dtype, keepdim=True)
        dist.all_reduce(group_sum)
        avg = (group_sum / nelements).to(x.dtype)

        group_var_sum = torch.empty((x.shape[0], self.num_groups), dtype=x.dtype, device=x.device)
        if rank % 2 == 0:
            torch.var(x[..., :-1], dim=(2, 3, 4, 5), out=group_var_sum, keepdim=True)
        else:
            torch.var(x[..., 1:], dim=(2, 3, 4, 5), out=group_var_sum, keepdim=True)
        group_var_sum = group_var_sum * (nelements_rank - 1)
        dist.all_reduce(group_var_sum)
        var = (group_var_sum / (nelements - 1)).to(x.dtype)

        x = (x - avg) / torch.sqrt(var + self.eps)
        x = x.view(x.shape[0], -1, *x.shape[3:])
        x = x * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]
        return x
