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
import torch_npu
import torch.nn as nn

sys.path.append('../')
from mindiesd import Linear

OUT_FEATURES = 1152
IN_FEATURES = 1280
BATCH = 2
SEQLEN = 16
HIDDEN_SIZE = 1152


class TestLinearModes(unittest.TestCase):
    def setUp(self):
        self.q = nn.Linear(OUT_FEATURES, IN_FEATURES).to("npu").to(torch.float16)
        self.x = torch.randn(BATCH, SEQLEN, HIDDEN_SIZE, dtype=torch.float16).to("npu")
        self.result = self.q(self.x)
        self.result_cpu = self.result.to("cpu").to(dtype=torch.float32).reshape(1, -1)

    def test_linear_bmmv3(self):
        self.q_sd = Linear(OUT_FEATURES, IN_FEATURES, op_type="batchmatmulv3").to("npu").to(torch.float16)
        self.q_sd.weight.data.copy_(self.q.weight.data)
        self.q_sd.bias.data.copy_(self.q.bias.data)
        self.q_sd.weight.transpose(0, 1)
        result_sd = self.q_sd(self.x)
        result_sd_cpu = result_sd.to("cpu").to(dtype=torch.float32).reshape(1, -1)

        cosine_sim = torch.cosine_similarity(self.result_cpu, result_sd_cpu)[0]

        delta = (self.result_cpu - result_sd_cpu).abs()
        max_error = delta.max().item()
        mean_error = delta.mean().item()

        self.assertGreater(cosine_sim, 0.999, f"Cosine similarity too low.")

    def test_linear_bmmv3_no_t(self):
        self.q_sd = Linear(OUT_FEATURES, IN_FEATURES, op_type="batchmatmulv3").to("npu").to(torch.float16)
        self.q_sd.weight.data.copy_(self.q.weight.data)
        self.q_sd.bias.data.copy_(self.q.bias.data)
        result_sd = self.q_sd(self.x)
        result_sd_cpu = result_sd.to("cpu").to(dtype=torch.float32).reshape(1, -1)

        cosine_sim = torch.cosine_similarity(self.result_cpu, result_sd_cpu)[0]

        delta = (self.result_cpu - result_sd_cpu).abs()
        max_error = delta.max().item()
        mean_error = delta.mean().item()

        self.assertGreater(cosine_sim, 0.999, f"Cosine similarity too low.")

    def test_linear_mmv2(self):
        self.q_sd = Linear(OUT_FEATURES, IN_FEATURES, op_type="matmulv2").to("npu").to(torch.float16)
        self.q_sd.weight.data.copy_(self.q.weight.data)
        self.q_sd.bias.data.copy_(self.q.bias.data)
        self.q_sd.weight.transpose(0, 1)
        result_sd = self.q_sd(self.x)
        result_sd_cpu = result_sd.to("cpu").to(dtype=torch.float32).reshape(1, -1)

        cosine_sim = torch.cosine_similarity(self.result_cpu, result_sd_cpu)[0]

        delta = (self.result_cpu - result_sd_cpu).abs()
        max_error = delta.max().item()
        mean_error = delta.mean().item()

        self.assertGreater(cosine_sim, 0.999, f"Cosine similarity too low.")

    def test_linear_mmv2_no_t(self):
        self.q_sd = Linear(OUT_FEATURES, IN_FEATURES, op_type="matmulv2").to("npu").to(torch.float16)
        self.q_sd.weight.data.copy_(self.q.weight.data)
        self.q_sd.bias.data.copy_(self.q.bias.data)
        result_sd = self.q_sd(self.x)
        result_sd_cpu = result_sd.to("cpu").to(dtype=torch.float32).reshape(1, -1)

        cosine_sim = torch.cosine_similarity(self.result_cpu, result_sd_cpu)[0]

        delta = (self.result_cpu - result_sd_cpu).abs()
        max_error = delta.max().item()
        mean_error = delta.mean().item()

        self.assertGreater(cosine_sim, 0.999, f"Cosine similarity too low.")

    def test_linear_bmmv2(self):
        self.q_sd = Linear(OUT_FEATURES, IN_FEATURES, op_type="batchmatmulv2").to("npu").to(torch.float16)
        self.q_sd.weight.data.copy_(self.q.weight.data)
        self.q_sd.bias.data.copy_(self.q.bias.data)
        self.q_sd.weight.transpose(0, 1)
        result_sd = self.q_sd(self.x)
        result_sd_cpu = result_sd.to("cpu").to(dtype=torch.float32).reshape(1, -1)

        cosine_sim = torch.cosine_similarity(self.result_cpu, result_sd_cpu)[0]

        delta = (self.result_cpu - result_sd_cpu).abs()
        max_error = delta.max().item()
        mean_error = delta.mean().item()

        self.assertGreater(cosine_sim, 0.999, f"Cosine similarity too low.")

    def test_linear_bmmv2_no_t(self):
        self.q_sd = Linear(OUT_FEATURES, IN_FEATURES, op_type="batchmatmulv2").to("npu").to(torch.float16)
        self.q_sd.weight.data.copy_(self.q.weight.data)
        self.q_sd.bias.data.copy_(self.q.bias.data)
        result_sd = self.q_sd(self.x)
        result_sd_cpu = result_sd.to("cpu").to(dtype=torch.float32).reshape(1, -1)

        cosine_sim = torch.cosine_similarity(self.result_cpu, result_sd_cpu)[0]

        delta = (self.result_cpu - result_sd_cpu).abs()
        max_error = delta.max().item()
        mean_error = delta.mean().item()

        self.assertGreater(cosine_sim, 0.999, f"Cosine similarity too low.")


if __name__ == "__main__":
    unittest.main()