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
import torch.nn as nn
from torch import Tensor
from .matmul import matmul_forward, batchmatmul_forward
from ..utils.exception import ParametersInvalid


class Linear(nn.Module):

    def __init__(
            self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None,
            op_type="matmulv2") -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.op_type = op_type

    def forward(self, input: Tensor) -> Tensor:
        if input.dim() < 2:
            raise ParametersInvalid("The input dimension should be greater than 1.")
        flattened_input, transpose_x2, original_shape = self._process_input(input)

        result = self._apply_matmul(flattened_input, self.weight, self.bias, self.op_type, transpose_x2)

        # If the input was originally not a 2D tensor, restore its original shape
        if original_shape is not None:
            result = result.view(*original_shape, -1)
        return result

    def _process_input(self, input_tensor: torch.Tensor):
        """
        Process the input tensor:
        - If it is a 3D tensor and "batch" is not in op_type, flatten it to 2D.
        - Return the flattened tensor and the original shape information.
        """
        input_shape = input_tensor.shape
        weight_shape = self.weight.shape

        input_last_dim = input_shape[-1]
        weight_last_dim = weight_shape[-1]
        transpose_x2 = (input_last_dim == weight_last_dim)

        if (input_tensor.ndim != 2) and ("batch" not in self.op_type):
            flattened_input = input_tensor.view(-1, input_last_dim)
            return flattened_input, transpose_x2, input_shape[:-1]
        else:
            return input_tensor, transpose_x2, None

    def _apply_matmul(self, input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, op_type: str,
                      transpose_x2: bool):
        """
        Apply matrix multiplication:
        - Depending on `op_type` and `transpose_x2`, call different implementations.
        """
        if "batch" in op_type:
            return batchmatmul_forward(input_tensor, weight, bias, transpose_x2=transpose_x2, op_type=op_type)
        else:
            return matmul_forward(input_tensor, weight, bias, transpose_x2=transpose_x2, op_type=op_type)
