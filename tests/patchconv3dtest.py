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

import os
import logging
import random
import math
from typing import Optional, Tuple, Union
import torch
import torch_npu
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.utils import _triple, _reverse_repeat_tuple
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_3_t

from tests.utils.utils.parallel_mgr import init_parallel_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatchConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        transposed: bool = False,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        block_size: Union[int, Tuple[int, int]] = 2,
        is_casual: bool = False,
        is_overlap: bool = True
    ) -> None:
        self.padding = padding if isinstance(padding, str) else _triple(padding)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.block_size = block_size
        self.is_casual = is_casual
        self.is_overlap = is_overlap
        self.rank = 0
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(self.kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, self.kernel_size,
                                   range(len(self.kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        # initialize weight and bias
        if transposed:
            self.weight = Parameter(torch.empty(
                (in_channels, out_channels // groups, *self.kernel_size), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels, in_channels // groups, *self.kernel_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        ch_in, ch_out, *_ = self.weight.shape
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = math.prod([item for item in self.kernel_size]) * ch_out
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
        
    def forward(self, patch_hidden_state: Tensor, weight: Tensor = None, bias: Tensor = None) -> Tensor:
        if weight is None:
            return self._conv_forward(patch_hidden_state, self.weight, self.bias)
        else:
            return self._conv_forward(patch_hidden_state, weight, bias)
    
    def _one_worldsize_conv(self, padding_mode, patch_hidden_state, weight, bias):
        if padding_mode != 'zeros':
            return F.conv3d(F.pad(patch_hidden_state, self._reversed_padding_repeated_twice, 
                                    mode=padding_mode), weight, bias, self.stride, 
                                    _triple(0), self.dilation, self.groups)
        return F.conv3d(patch_hidden_state, weight, bias, self.stride, 
                        self.padding, self.dilation, self.groups)

    def _pre_conv_forward(self, patch_hidden_state, shape):
        bs, channels, t, h, _ = shape
        if self.rank % 2 == 0 and self.rank != 0:
            send = patch_hidden_state[..., :1].contiguous()
            send_op = dist.P2POp(dist.isend, send, self.rank - 1)
            recv = torch.zeros([bs, channels, t, h, 1], 
                dtype=patch_hidden_state.dtype, device=f"npu:{self.rank}")
            recv_op = dist.P2POp(dist.irecv, recv, self.rank - 1)
            dist.batch_isend_irecv([send_op, recv_op])
            return recv
        elif self.rank % 2 != 0 and self.rank != self.world_size - 1:
            send = patch_hidden_state[..., -1:].contiguous()
            send_op = dist.P2POp(dist.isend, send, self.rank + 1)
            recv = torch.zeros([bs, channels, t, h, 1], 
                dtype=patch_hidden_state.dtype, device=f"npu:{self.rank}")
            recv_op = dist.P2POp(dist.irecv, recv, self.rank + 1)
            dist.batch_isend_irecv([send_op, recv_op])
            return recv
        return None
        

    def _end_conv_forward(self, outputs, shape):  
        bs_, channels_, t_, h_, _ = shape
        if self.rank % 2 == 0:
            send = outputs[0][..., -1:].contiguous()
            send_op = dist.P2POp(dist.isend, send, self.rank + 1)
            recv = torch.zeros([bs_, channels_, t_, h_, 1], 
                dtype=outputs[0].dtype, device=f"npu:{self.rank}")
            recv_op = dist.P2POp(dist.irecv, recv, self.rank + 1)
            dist.batch_isend_irecv([send_op, recv_op])
        else:
            send = outputs[0][..., :1].contiguous()
            send_op = dist.P2POp(dist.isend, send, self.rank - 1)
            recv = torch.zeros([bs_, channels_, t_, h_, 1], 
                dtype=outputs[0].dtype, device=f"npu:{self.rank}")
            recv_op = dist.P2POp(dist.irecv, recv, self.rank - 1)
            dist.batch_isend_irecv([send_op, recv_op])
        return recv

    def _parallel_conv_forward(self, patch_hidden_state, weight, bias):
        shape = patch_hidden_state.shape
        bs, channels, t, h, w = shape
        patch_hidden_state, padding = self._adjust_padding_for_patch(patch_hidden_state, self.padding)
        stride = (w - 1 + self.block_size - 1) // self.block_size
        overlap = self.kernel_size[0] // 2
        outputs = []
        recv = None
        # P2P communication
        for step in range(self.block_size):
            start_idx = step * stride + 1 - overlap
            end_idx = min((step + 1) * stride + 1 + overlap, w)
            if self.rank % 2 == 0:
                input_patch = patch_hidden_state[..., w - end_idx:w - start_idx]
            else:
                input_patch = patch_hidden_state[..., start_idx:end_idx]

            if step == 0:
                recv = self._pre_conv_forward(patch_hidden_state, shape)
            if step == self.block_size - 1:
                if overlap == 1:
                    input_patch = torch.cat([recv, input_patch], dim=-1) \
                        if self.rank % 2 == 0 else torch.cat([input_patch, recv], dim=-1)
                recv = self._end_conv_forward(outputs, outputs[0].shape)
            
            outputs.append(F.conv3d(input_patch, weight, bias, self.stride, padding, self.dilation, self.groups))

            if step == 0:
                if self.rank == 0:
                    recv = torch.zeros([bs, channels, t, h, 1],
                        dtype=patch_hidden_state.dtype, device=f"npu:{self.rank}")
                elif self.rank == self.world_size - 1:
                    recv = torch.zeros([bs, channels, t, h, 1],
                        dtype=patch_hidden_state.dtype, device=f"npu:{self.rank}")
            if step == self.block_size - 1:
                if self.rank % 2 == 0:
                    outputs.insert(0, recv)
                    outputs.reverse()
                else:
                    outputs.insert(0, recv)

        return torch.cat(outputs, dim=-1)

    def _conv_forward(self, patch_hidden_state: Tensor, weight: Tensor, bias: Optional[Tensor]):
        self._get_world_size_and_rank()
        if (self.world_size == 1):
            return self._one_worldsize_conv(self.padding_mode, patch_hidden_state, weight, bias)
        else:
            return self._parallel_conv_forward(patch_hidden_state, weight, bias)
            
    def _get_world_size_and_rank(self):
        world_size = 1
        rank = 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        self.world_size = world_size
        self.rank = rank
    
    def _adjust_padding_for_patch(self, patch_input, padding):
        if self.kernel_size[-1] == 3 and self.is_casual:
            patch_input = patch_input[..., 1:-1]
        padding = list(padding)
        padding[-1] = 0
        return patch_input, tuple(padding)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def speed_conv3d(hidden_state, rank, convs):
    # speed test for nn.Conv3d
    custom_stream = torch_npu.npu.Stream()
    for i in range(20):
        with torch_npu.npu.stream(custom_stream):
            start_event = torch_npu.npu.Event(enable_timing=True)
            end_event = torch_npu.npu.Event(enable_timing=True)
            start_event.record()
            result = convs(hidden_state)
            end_event.record()
            custom_stream.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            if rank == 0:
                logger.info("Round:%f, time: %f ms", i, elapsed_time)


class Patchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def forward(self, hidden_state, dim, is_overlap):
        length = hidden_state.shape[dim]
        if is_overlap:
            overlap = self.rank % 2
            start_idx = (length + self.world_size - 1) // self.world_size * self.rank - overlap
            end_idx = min((length + self.world_size - 1) // self. world_size * (self.rank + 1) - overlap + 1, length)
        else:
            start_idx = (length + self.world_size - 1) // self.world_size * self.rank
            end_idx = min((length + self.world_size - 1) // self.world_size * (self.rank + 1), length)
        idx = torch.arange(start_idx, end_idx, device=f"npu:{self.rank}")
        return hidden_state.index_select(dim, idx).clone()
        

class Depatchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def forward(self, patch_hidden_state, dim, is_overlap):
        if is_overlap:
            overlap = self.rank % 2
            start_idx = overlap
            end_idx = patch_hidden_state.shape[dim] + overlap - 1
            idx = torch.arange(start_idx, end_idx, device=f"npu:{self.rank}")
            patch_hidden_state = patch_hidden_state.index_select(dim, idx)
        
        patch_length_list = [torch.empty([1], dtype=torch.int64, device=f"npu:{self.rank}") 
                             for _ in range(self.world_size)]
        dist.all_gather(
            patch_length_list,
            torch.tensor(
                [patch_hidden_state.shape[dim]],
                dtype=torch.int64,
                device=f"npu:{self.rank}"
            )
        )
        patch_shape = list(patch_hidden_state.shape)
        patch_hidden_state_list = []
        for i in range(self.world_size):
            patch_shape[dim] = patch_length_list[i].item()
            patch_hidden_state_list.append(
                torch.empty(tuple(patch_shape), dtype=patch_hidden_state.dtype, device=f"npu:{self.rank}"))
        dist.all_gather(
            patch_hidden_state_list,
            patch_hidden_state.contiguous()
        )

        return torch.cat(patch_hidden_state_list, dim)


def speed_patchconv3d(hidden_state, rank, patch_convs):
    # speed test for PatchConv3d
    patch = Patchify()
    depatch = Depatchify()
    custom_stream = torch_npu.npu.Stream()
    for i in range(20):
        with torch_npu.npu.stream(custom_stream):
            start_event = torch_npu.npu.Event(enable_timing=True)
            end_event = torch_npu.npu.Event(enable_timing=True)
            patch_hidden_state = patch(hidden_state, dim=-1, is_overlap=True)
            start_event.record()
            patch_result = patch_convs(patch_hidden_state)
            end_event.record()
            custom_stream.synchronize()
            patch_result = depatch(patch_result, dim=-1, is_overlap=True)
            elapsed_time = start_event.elapsed_time(end_event)
            if rank == 0:
                logger.info("Round:%f, time: %f ms", i, elapsed_time)


def performance_test(hidden_state, rank, convs, patch_convs):
    # performance test
    result = convs(hidden_state)
    patch = Patchify()
    depatch = Depatchify()
    patch_hidden_state = patch(hidden_state, dim=-1, is_overlap=True)
    patch_hidden_state = patch_convs(patch_hidden_state, convs.weight, convs.bias)
    patch_result = depatch(patch_hidden_state, dim=-1, is_overlap=True)
    if rank == 0:
        if not torch.allclose(result, patch_result, atol=1e-10):
            raise ValueError("This method is only available when cal_loss is True")


def main():
    set_seed()
    init_parallel_env(True)
    rank = dist.get_rank()
    torch.device('npu', rank)

    convs = nn.Conv3d(256, 512, 3, 1, 1).to(f"npu:{rank}", dtype=torch.bfloat16)
    patch_convs = PatchConv3d(256, 512, 3, 1, 1).to(f"npu:{rank}", dtype=torch.bfloat16)
    hidden_state = torch.randn(
        1, 256, 6, 92, 162, dtype=torch.bfloat16, device=f"npu:{rank}")
    performance_test(hidden_state, rank, convs, patch_convs)
    speed_conv3d(hidden_state, rank, convs)
    speed_patchconv3d(hidden_state, rank, patch_convs)



if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    main()