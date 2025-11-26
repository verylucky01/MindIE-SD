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

import unittest
from unittest.mock import patch

import sys
import time

import torch

sys.path.append('../')

from device import DEVICE_ID
from mindiesd.layers.flash_attn.common import AttentionParam
from mindiesd.layers.flash_attn.attention_forward import attention_forward


class TestAttentionFunc(unittest.TestCase):
    def test_attn_forward_no_fused(self):
        attention_shape = [2, 32, 16, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out = attention_forward(query, key, value, fused=False)
        self.assertIsNotNone(out)

    def test_attn_forward_runtime(self):
        attention_shape = [2, 32, 16, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out = attention_forward(query, key, value, opt_mode="runtime")
        self.assertIsNotNone(out)

    def test_attn_forward_static(self):
        attention_shape = [2, 32, 16, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out = attention_forward(query, key, value, opt_mode="static")
        self.assertIsNotNone(out)

    def test_attn_forward_manual(self):
        attention_shape = [2, 32, 16, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out = attention_forward(query, key, value, opt_mode="manual", op_type="prompt_flash_attn", layout="BNSD")
        self.assertIsNotNone(out)

    def test_attn_forward_manual_la(self):
        attention_shape = [2, 5120, 16, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out = attention_forward(query, key, value, opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
        self.assertIsNotNone(out)


if __name__ == '__main__':
    import torch_npu

    torch_npu.npu.set_device(DEVICE_ID)
    unittest.main()