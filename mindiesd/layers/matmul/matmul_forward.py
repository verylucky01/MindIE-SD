#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
from pathlib import Path

import torch
from ...utils.exception import ParametersInvalid
from ...utils import logger, file_utils
from .. import _custom_ops as ops



def matmul_forward(x1, x2, bias=None, transpose_x1=False, transpose_x2=False, offset_x=0, offset_w=None,
                   enable_hf32=False, op_type="matmulv2"):
    """
    参数:
        x1 (Tensor): 输入张量1。
        x2 (Tensor): 输入张量2。
        bias (Tensor, optional): 偏置张量，默认为 None。
        transpose_x1 (bool, optional): 是否转置 x1, 默认为 False。
        transpose_x2 (bool, optional): 是否转置 x2, 默认为 False。
        offset_x (int, optional): 输入偏移量，默认为 0。
        enable_hf32 (bool, optional): 是否启用 hf32, 默认为 False(仅 v3 支持）。
        offset_w (Tensor, optional): 权重偏移量，默认为 None(仅 v2 支持）。
        op_type (str, optional): 目前仅支持 "matmulv2"。

    返回:
        Tensor: 计算结果。
    """
    if op_type != "matmulv2":
        logger.warning(f"Unsupported op_type: {op_type}. Currently only use 'matmulv2', 'matmulv3' needs to be fixed.")
    
    return _matmulv2_forward(x1, x2, bias, transpose_x1, transpose_x2, offset_x, offset_w)


def _matmulv2_forward(x1, x2, bias=None, transpose_x1=False, transpose_x2=False,
                      offset_x=0, offset_w=None):
    return ops.matmul_v2(
        input_x1=x1,
        input_x2=x2,
        bias=bias,
        offset_w=offset_w,
        transpose_x1=transpose_x1,
        transpose_x2=transpose_x2,
        offset_x=offset_x
    )