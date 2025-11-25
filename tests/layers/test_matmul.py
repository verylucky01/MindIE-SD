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
import sys
import torch
import torch.nn.functional as F

sys.path.append('../')
from mindiesd.layers.matmul.matmul_forward import matmul_forward


class TestMatMulForward(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)

        self.m = 75888
        self.k = 6144
        self.n = 9216

        self.x1 = torch.randn((self.m, self.k), device=self.device, dtype=torch.float16)
        self.x2 = torch.randn((self.k, self.n), device=self.device, dtype=torch.float16)

        self.reference_result = torch.matmul(self.x1, self.x2)

    def test_matmul_forward_other(self):
        y_matmulv2 = matmul_forward(self.x1, self.x2, op_type="111")

        y_matmulv2_cpu = y_matmulv2.to("cpu").to(dtype=torch.float32).reshape(1, -1)
        reference_cpu = self.reference_result.to("cpu").to(dtype=torch.float32).reshape(1, -1)

        cosine_sim = torch.cosine_similarity(y_matmulv2_cpu, reference_cpu)[0]

        self.assertGreater(cosine_sim.item(), 0.99, "Cosine similarity is too low for op_type v3!")

        delta = (y_matmulv2_cpu - reference_cpu).abs()
        max_error = delta.max().item()
        mean_error = delta.mean().item()

    def test_matmul_forward_v2(self):
        y_matmulv2 = matmul_forward(self.x1, self.x2, op_type="v2")

        y_matmulv2_cpu = y_matmulv2.to("cpu").to(dtype=torch.float32).reshape(1, -1)
        reference_cpu = self.reference_result.to("cpu").to(dtype=torch.float32).reshape(1, -1)

        cosine_sim = torch.cosine_similarity(y_matmulv2_cpu, reference_cpu)[0]

        self.assertGreater(cosine_sim.item(), 0.99, "Cosine similarity is too low for op_type v2!")

        delta = (y_matmulv2_cpu - reference_cpu).abs()
        max_error = delta.max().item()
        mean_error = delta.mean().item()


if __name__ == "__main__":
    unittest.main()