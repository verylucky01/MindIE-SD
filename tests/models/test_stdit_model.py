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

import sys
import unittest
from unittest.mock import patch
sys.path.append('../')

import torch
from device import DEVICE_ID
from mindiesd.models import STDiT3, STDiT3Config
from mindiesd.utils import ModelInitError

torch.npu.set_device(DEVICE_ID)
NPU = "npu"


class TestSTDiT3(unittest.TestCase):
    def test_init(self):
        with patch.dict('sys.modules', {'opensora': None}):
            with self.assertRaises(ModelInitError) as context:
                STDiT3(STDiT3Config())
            self.assertIn("Failed to find the opensora library", str(context.exception))


if __name__ == '__main__':
    unittest.main()