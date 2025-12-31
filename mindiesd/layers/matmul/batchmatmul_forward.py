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
from ...utils.get_platform import get_npu_device, NPUDevice
from ...utils.exception import ParametersInvalid
from ...utils import logger, file_utils

BATCHMATMUL_V2 = "batchmatmulv2"
BATCHMATMUL_V3 = "batchmatmulv3"

current_path = Path(__file__).resolve()
if len(current_path.parents) < 3:
    raise ParametersInvalid("The parents level is insufficient.")
ops_path = current_path.parents[2] / "plugin"
ops_path = file_utils.standardize_path(str(ops_path))
ops_file = os.path.join(ops_path, "libPTAExtensionOPS.so")
file_utils.check_file_safety(ops_file, permission_mode=file_utils.BINARY_FILE_PERMISSION)
torch.ops.load_library(ops_file)


def batchmatmul_forward(x1, x2, bias=None, transpose_x1=False, transpose_x2=False, offset_x=0, offset_w=None,
                        enable_hf32=False, op_type=BATCHMATMUL_V2):
    """
    统一接口封装 batchmatmulv3_mindie_sd、batchmatmulv3duo_mindie_sd 和 batchmatmulv2_mindie_sd。

    参数:
        x1 (Tensor): 输入张量1。
        x2 (Tensor): 输入张量2。
        bias (Tensor, optional): 偏置张量，默认为 None。
        offset_w (Tensor, optional): 权重偏移量，默认为 None。
        adj_x1 (bool, optional): 是否转置 x, 默认为 False。
        adj_x2 (bool, optional): 是否转置 x2, 默认为 False。
        offset_x (int, optional): 输入偏移量，默认为 0。
        enable_hf32 (bool, optional): 是否启用 hf32, 默认为 False(仅 v3 支持）。
        op_type (str, optional): 使用的版本 ("batchmatmulv2"、"batchmatmulv3")，默认为 "batchmatmulv2"。
    返回:
        Tensor: 计算结果。
    """
    if op_type not in [BATCHMATMUL_V2, BATCHMATMUL_V3]:
        logger.warning(f"Unsupported op_type: {op_type}."
                       f"Please use 'batchmatmulv2', 'batchmatmulv3'. Defaulting to 'batchmatmulv2'.")
        op_type = BATCHMATMUL_V2

    if op_type == BATCHMATMUL_V3:
        return _batchmatmulv3_forward(x1, x2, bias, transpose_x1, transpose_x2, offset_x, offset_w, enable_hf32)
    else:
        return _batchmatmulv2_forward(x1, x2, bias, transpose_x1, transpose_x2, offset_x, offset_w, enable_hf32)


def _batchmatmulv2_forward(x1, x2, bias=None, transpose_x1=False, transpose_x2=False,
                           offset_x=0, offset_w=None, enable_hf32=False):
    if enable_hf32:
        logger.warning("The enable_hf32 parameter is not supported in batchmatmulv2_mindie_sd and will be ignored.")

    # use batchmatmulv2_mindie_sd
    return torch.ops.mindie.batchmatmulv2_mindie_sd(
        input_x1=x1,
        input_x2=x2,
        bias=bias,
        offset_w=offset_w,
        adj_x1=transpose_x1,
        adj_x2=transpose_x2,
        offset_x=offset_x
    )


def _batchmatmulv3_forward(x1, x2, bias=None, transpose_x1=False, transpose_x2=False,
                           offset_x=0, offset_w=None, enable_hf32=False):
    npu_device = get_npu_device()
    if npu_device == NPUDevice.A2:
        # use batchmatmulv3_mindie_sd
        return torch.ops.mindie.batchmatmulv3_mindie_sd(
            x1=x1,
            x2=x2,
            bias=bias,
            offset_w=offset_w,
            adj_x1=transpose_x1,
            adj_x2=transpose_x2,
            offset_x=offset_x,
            enable_hf32=enable_hf32
        )
    elif npu_device == NPUDevice.Duo:
        unspoorted_params = transpose_x1 or transpose_x2 or offset_x != 0 or enable_hf32
        if unspoorted_params:
            logger.warning("batchmatmulv3duo_mindie_sd only supports the x1, x2, bias, \
                            and offset_w parameters. Other parameters will be ignored.")

        # use batchmatmulv3duo_mindie_sd
        return torch.ops.mindie.batchmatmulv3duo_mindie_sd(
            x1=x1,
            x2=x2,
            bias=bias,
            offset_w=offset_w
        )
    else:
        raise ParametersInvalid("Platform invalid. Please check env.")
