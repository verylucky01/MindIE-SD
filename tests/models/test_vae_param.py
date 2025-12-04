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
from unittest.mock import MagicMock

import torch

from mindiesd.utils import ParametersInvalid, ConfigError
from mindiesd.models.vae import check_init_params, check_get_latent_size, check_decode

sys.path.append('../')


class TestCheckInitParams(unittest.TestCase):

    def setUp(self):
        self.mock_config = MagicMock()

    @check_init_params
    def init_func(self, config):
        pass

    def test_invalid_param_types(self):
        self.mock_config.vae_2d = ["from_pretrained", "subfolder", "micro_batch_size"]
        self.mock_config.freeze_vae_2d = "False"
        self.mock_config.micro_frame_size = "17"
        self.mock_config.shift = [-0.10, 0.34, 0.27, 0.98]
        self.mock_config.scale = "3.85, 2.32, 2.33, 3.06"
        self.mock_config.set_patch_parallel = "False"

        with self.assertRaises(ConfigError):
            self.init_func(self.mock_config)


class TestCheckGetLatentSize(unittest.TestCase):

    @check_get_latent_size
    def forward_func(self, input_size):
        pass

    def test_valid_parameters(self):
        input_size = (32, 720, 1280)

        try:
            self.forward_func(input_size)
        except ParametersInvalid:
            self.fail("Invalid parameters raised unexpectedly!")

    def test_invalid_input_size_type(self):
        input_size = [32, 720, 1280]

        with self.assertRaises(ParametersInvalid):
            self.forward_func(input_size)

    def test_invalid_input_size_dimension(self):
        input_size = (720, 1280)

        with self.assertRaises(ParametersInvalid):
            self.forward_func(input_size)

    def test_wrong_input_size_value(self):
        input_size = (0, -720, -1280)

        with self.assertRaises(ParametersInvalid):
            self.forward_func(input_size) 


class TestCheckDecode(unittest.TestCase):

    @check_decode
    def forward_func(self, z, num_frames):
        pass

    def test_valid_parameters(self):
        z = torch.randn(1, 4, 9, 90, 160)
        num_frames = 32

        try:
            self.forward_func(z, num_frames)
        except ParametersInvalid:
            self.fail("Invalid parameters raised unexpectedly!")

    def test_invalid_z_type(self):
        z = (1, 4, 9, 90, 160)
        num_frames = 32

        with self.assertRaises(ParametersInvalid):
            self.forward_func(z, num_frames)

    def test_invalid_num_frames_type(self):
        z = torch.randn(1, 4, 9, 90, 160)
        num_frames = "32"

        with self.assertRaises(ParametersInvalid):
            self.forward_func(z, num_frames)

    def test_wrong_z_dimension(self):
        z = torch.randn(1, 4, 9, 90)
        num_frames = 32

        with self.assertRaises(ParametersInvalid):
            self.forward_func(z, num_frames) 

    def test_wrong_num_frames_value(self):
        z = torch.randn(1, 4, 9, 90, 160)
        num_frames = -32

        with self.assertRaises(ParametersInvalid):
            self.forward_func(z, num_frames) 


if __name__ == '__main__':
    unittest.main()