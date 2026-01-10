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
import torch
import torch_npu
sys.path.append('../')

from device import DEVICE_ID
from mindiesd.layers.flash_attn.sparse_flash_attn import sparse_attention
from tests.utils.utils.precision_compare import data_compare


class TestSparseAttention(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        self.batch_size = 1
        self.head = 3
        self.q_seqlen = 9600
        self.kv_seqlen = 9600
        self.headdim = 128
        self.scale = self.headdim ** -0.5
        self.t, self.h, self.w = 3, 40, 80

        q_shape = (self.batch_size, self.q_seqlen, self.head, self.headdim)
        kv_shape = (self.batch_size, self.kv_seqlen, self.head, self.headdim)
        self.q = torch.randn(q_shape, dtype=torch.float16, device=self.device)
        self.k = torch.randn(kv_shape, dtype=torch.float16, device=self.device)
        self.v = torch.randn(kv_shape, dtype=torch.float16, device=self.device)

    def test_rf_v2(self):
        out = sparse_attention(
            self.q, self.k, self.v,
            scale=self.scale,
            head_num=self.head,
            input_layout="BSND",
            inner_precise=0,
            sparse_type="rf_v2",
            txt_len=0,
            latent_shape_q=(self.t, self.h, self.w),
            latent_shape_k=(self.t, self.h, self.w),
            sparsity=0.0
        )
        self.assertIsNotNone(out)

    def test_ada_bsa(self):
        out = sparse_attention(
            self.q, self.k, self.v,
            scale=self.scale,
            head_num=self.head,
            input_layout="BSND",
            inner_precise=0,
            sparse_type="ada_bsa",
            cdf_threshold=1.0,
            sparsity=0.0
        )
        self.assertIsNotNone(out)
    
    def test_rf_v2_result(self):
        out = sparse_attention(
            self.q, self.k, self.v,
            scale=self.scale,
            head_num=self.head,
            input_layout="BSND",
            inner_precise=0,
            sparse_type="rf_v2",
            txt_len=0,
            latent_shape_q=(self.t, self.h, self.w),
            latent_shape_k=(self.t, self.h, self.w),
            sparsity=0.0
        )
        fascore = torch_npu.npu_fusion_attention(
                    self.q, self.k, self.v,
                    input_layout="BSND",
                    scale=self.scale,
                    pre_tockens=2147483647,
                    next_tockens=2147483647,
                    head_num=self.head)[0]

        result, _, max_err = data_compare(out.cpu(), fascore.cpu())
        self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")

    def test_ada_bsa_result(self):
        out = sparse_attention(
            self.q, self.k, self.v,
            scale=self.scale,
            head_num=self.head,
            input_layout="BSND",
            inner_precise=0,
            sparse_type="ada_bsa",
            cdf_threshold=1.0,
            sparsity=0.0
        )
        fascore = torch_npu.npu_fusion_attention(
                    self.q, self.k, self.v,
                    input_layout="BSND",
                    scale=self.scale,
                    pre_tockens=2147483647,
                    next_tockens=2147483647,
                    head_num=self.head)[0]
        result, _, max_err = data_compare(out.cpu(), fascore.cpu())
        self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")


if __name__ == '__main__':
    torch_npu.npu.set_device(DEVICE_ID)
    unittest.main()