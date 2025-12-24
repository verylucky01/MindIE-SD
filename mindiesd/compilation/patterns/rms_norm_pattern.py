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
from ..passes.register_pattern_to_pass import PatternBase

if hasattr(torch.npu, "is_available"):
    npu_available = torch.npu.is_available()
if npu_available:
    import torch_npu
    import mindiesd


def create(dtype):
    class RMSNormPattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            hidden_states = torch.empty(2, 2, dtype=dtype, device="meta")
            weight = torch.empty(2, dtype=dtype, device="meta")
            return [hidden_states, weight]

        @staticmethod
        def pattern(query, dim_head):
            def func(query, dim_head):
                norm_q = torch.nn.RMSNorm(dim_head, eps=1e-6)
                return norm_q(query)
            return func(query, dim_head)

        @staticmethod
        def replacement(query, dim_head):
            def func(query, dim_head):
                norm_q = mindiesd.RMSNorm(dim_head, eps=1e-6)
                return norm_q(query)
            return func(query, dim_head)
    return RMSNormPattern

RMSNormPatternGroup = [create(torch.bfloat16)]
