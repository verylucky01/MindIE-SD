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

sys.path.append('../')
from mindiesd.utils import ConfigError, ParametersInvalid
from mindiesd.models.stdit3 import check_init_params, check_forward_params


class TestCheckInitParams(unittest.TestCase):

    def setUp(self):
        self.mock_config = MagicMock()

    @check_init_params
    def init_func(self, config):
        pass

    def test_incorrect_param_types(self):
        self.mock_config.input_size = [1, 2]
        self.mock_config.in_channels = '3'
        self.mock_config.caption_channels = '4'
        self.mock_config.enable_flash_attn = 'True'
        self.mock_config.enable_sequence_parallelism = 'False'
        self.mock_config.use_cache = 'True'
        self.mock_config.cache_interval = '5'
        self.mock_config.cache_start = '6'
        self.mock_config.num_cache_layer = '7'
        self.mock_config.cache_start_steps = '8'

        with self.assertRaises(ConfigError):
            self.init_func(self.mock_config)


class TestCheckForwardParams(unittest.TestCase):

    @check_forward_params
    def forward_func(self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, t_idx=0):
        pass

    def test_valid_parameters(self):
        x = torch.randn(1, 2, 3, 4, 5)
        timestep = torch.randn(1)
        y = torch.randn(1, 2, 3, 4)
        mask = torch.randn(1, 2)
        x_mask = torch.randn(1, 2)
        fps = torch.tensor([30])
        height = torch.randn(1)
        width = torch.randn(1)
        t_idx = 0

        try:
            self.forward_func(x, timestep, y, mask, x_mask, fps, height, width, t_idx)
        except ParametersInvalid:
            self.fail("Invalid parameters raised unexpectedly!")

    def test_invalid_x_dimension(self):
        x = torch.randn(1, 2, 3, 4)
        timestep = torch.randn(1)
        y = torch.randn(1, 2, 3, 4)
        mask = torch.randn(1, 2)
        x_mask = torch.randn(1, 2)
        fps = torch.tensor([30])
        height = torch.randn(1)
        width = torch.randn(1)
        t_idx = 0

        with self.assertRaises(ParametersInvalid):
            self.forward_func(x, timestep, y, mask, x_mask, fps, height, width, t_idx)

    def test_invalid_timestep_dimension(self):
        x = torch.randn(1, 2, 3, 4, 5)
        timestep = torch.randn(1, 2)
        y = torch.randn(1, 2, 3, 4)
        mask = torch.randn(1, 2)
        x_mask = torch.randn(1, 2)
        fps = torch.tensor([30])
        height = torch.randn(1)
        width = torch.randn(1)
        t_idx = 0

        with self.assertRaises(ParametersInvalid):
            self.forward_func(x, timestep, y, mask, x_mask, fps, height, width, t_idx)

    def test_invalid_y_dimension(self):
        x = torch.randn(1, 2, 3, 4, 5)
        timestep = torch.randn(1)
        y = torch.randn(1, 2, 3)
        mask = torch.randn(1, 2)
        x_mask = torch.randn(1, 2)
        fps = torch.tensor([30])
        height = torch.randn(1)
        width = torch.randn(1)
        t_idx = 0

        with self.assertRaises(ParametersInvalid):
            self.forward_func(x, timestep, y, mask, x_mask, fps, height, width, t_idx)

    def test_invalid_mask_dimension(self):
        x = torch.randn(1, 2, 3, 4, 5)
        timestep = torch.randn(1)
        y = torch.randn(1, 2, 3, 4)
        mask = torch.randn(1, 2, 3)
        x_mask = torch.randn(1, 2)
        fps = torch.tensor([30])
        height = torch.randn(1)
        width = torch.randn(1)
        t_idx = 0

        with self.assertRaises(ParametersInvalid):
            self.forward_func(x, timestep, y, mask, x_mask, fps, height, width, t_idx)

    def test_invalid_x_mask_dimension(self):
        x = torch.randn(1, 2, 3, 4, 5)
        timestep = torch.randn(1)
        y = torch.randn(1, 2, 3, 4)
        mask = torch.randn(1, 2)
        x_mask = torch.randn(1, 2, 3)
        fps = torch.tensor([30])
        height = torch.randn(1)
        width = torch.randn(1)
        t_idx = 0

        with self.assertRaises(ParametersInvalid):
            self.forward_func(x, timestep, y, mask, x_mask, fps, height, width, t_idx)

    def test_invalid_height_dimension(self):
        x = torch.randn(1, 2, 3, 4, 5)
        timestep = torch.randn(1)
        y = torch.randn(1, 2, 3, 4)
        mask = torch.randn(1, 2)
        x_mask = torch.randn(1, 2)
        fps = torch.tensor([30])
        height = torch.randn(1, 2)
        width = torch.randn(1)
        t_idx = 0

        with self.assertRaises(ParametersInvalid):
            self.forward_func(x, timestep, y, mask, x_mask, fps, height, width, t_idx)

    def test_invalid_width_dimension(self):
        x = torch.randn(1, 2, 3, 4, 5)
        timestep = torch.randn(1)
        y = torch.randn(1, 2, 3, 4)
        mask = torch.randn(1, 2)
        x_mask = torch.randn(1, 2)
        fps = torch.tensor([30])
        height = torch.randn(1)
        width = torch.randn(1, 2)
        t_idx = 0

        with self.assertRaises(ParametersInvalid):
            self.forward_func(x, timestep, y, mask, x_mask, fps, height, width, t_idx)

    def test_invalid_fps_type(self):
        x = torch.randn(1, 2, 3, 4, 5)
        timestep = torch.randn(1)
        y = torch.randn(1, 2, 3, 4)
        mask = torch.randn(1, 2)
        x_mask = torch.randn(1, 2)
        fps = "30"
        height = torch.randn(1)
        width = torch.randn(1)
        t_idx = 0

        with self.assertRaises(ParametersInvalid):
            self.forward_func(x, timestep, y, mask, x_mask, fps, height, width, t_idx)

    def test_invalid_t_idx_type(self):
        x = torch.randn(1, 2, 3, 4, 5)
        timestep = torch.randn(1)
        y = torch.randn(1, 2, 3, 4)
        mask = torch.randn(1, 2)
        x_mask = torch.randn(1, 2)
        fps = torch.tensor([30])
        height = torch.randn(1)
        width = torch.randn(1)
        t_idx = "0"

        with self.assertRaises(ParametersInvalid):
            self.forward_func(x, timestep, y, mask, x_mask, fps, height, width, t_idx)


if __name__ == '__main__':
    unittest.main()
