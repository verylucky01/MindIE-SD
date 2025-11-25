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
from time import time
import logging
import torch
import torch.nn as nn


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

sys.path.append('../')
from mindiesd import Linear
OUT_FEATURES = 1152
IN_FEATURES = 1280
BATCH = 2
SEQLEN = 16
HIDDEN_SIZE = 1152


class TestLinearModes(unittest.TestCase):
    results = []  # 用于存储每个测试用例的执行时间和配置

    @classmethod
    def tearDownClass(cls):
        # 找出执行时间最短的算子及配置
        fastest_test = min(cls.results, key=lambda x: x[2])
        logging.info("\nExecution Times:")
        for test_name, op_type, execution_time in cls.results:
            logging.info(f"{test_name} ({op_type}): {execution_time:.6f} seconds")
        logging.info(
            f"\nFastest Operator: {fastest_test[0]} ({fastest_test[1]}) with time {fastest_test[2]:.6f} seconds")

    def setUp(self):

        self.q = nn.Linear(OUT_FEATURES, IN_FEATURES, device="npu", dtype=torch.bfloat16)
        self.x = torch.randn(BATCH, SEQLEN, HIDDEN_SIZE, device="npu", dtype=torch.float16)

    def run_test_and_measure_time(self, test_name, op_type, transpose_weight=True):
        """通用函数，用于运行测试并测量时间"""
        self.q_sd = Linear(OUT_FEATURES, IN_FEATURES, device="npu", dtype=torch.bfloat16, op_type=op_type)
        self.q_sd.weight.data.copy_(self.q.weight.data)
        self.q_sd.bias.data.copy_(self.q.bias.data)
        if transpose_weight:
            self.q_sd.weight.transpose(0, 1)
        start_time = time()
        result_sd = self.q_sd(self.x)
        end_time = time()

        execution_time = end_time - start_time
        self.results.append((test_name, op_type, execution_time))

        logging.info(f"{test_name} ({op_type}) completed in {execution_time:.6f} seconds.")

    def test_linear_bmmv3(self):
        self.run_test_and_measure_time("test_linear_bmmv3", "batchmatmulv3", transpose_weight=True)

    def test_linear_bmmv3_no_t(self):
        self.run_test_and_measure_time("test_linear_bmmv3_no_t", "batchmatmulv3", transpose_weight=False)

    def test_linear_mmv2(self):
        self.run_test_and_measure_time("test_linear_mmv2", "matmulv2", transpose_weight=True)

    def test_linear_mmv2_no_t(self):
        self.run_test_and_measure_time("test_linear_mmv2_no_t", "matmulv2", transpose_weight=False)

    def test_linear_bmmv2(self):
        self.run_test_and_measure_time("test_linear_bmmv2", "batchmatmulv2", transpose_weight=True)

    def test_linear_bmmv2_no_t(self):
        self.run_test_and_measure_time("test_linear_bmmv2_no_t", "batchmatmulv2", transpose_weight=False)


if __name__ == "__main__":
    unittest.main()