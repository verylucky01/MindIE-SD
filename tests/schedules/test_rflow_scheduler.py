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

import logging
import sys
import unittest
from unittest.mock import patch
from typing import List

import torch

sys.path.append('../')
from mindiesd.schedulers.rectified_flow import RFlowScheduler, validate_add_noise, validate_init_params, validate_step
from mindiesd.utils import ModelInitError, ParametersInvalid

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TestValidateInitParams(unittest.TestCase):
    @validate_init_params
    def init_func(self, num_timesteps, num_sampling_steps):
        pass

    def test_invalid_timesteps_type(self):
        num_timesteps = "1000"
        num_sampling_steps = 30

        with self.assertRaises(ModelInitError):
            self.init_func(num_timesteps, num_sampling_steps)

    def test_invalid_sampling_steps_type(self):
        num_timesteps = 1000
        num_sampling_steps = "30"

        with self.assertRaises(ModelInitError):
            self.init_func(num_timesteps, num_sampling_steps)

    def test_wrong_timesteps_nagative_value(self):
        num_timesteps = -1000
        num_sampling_steps = 30

        with self.assertRaises(ModelInitError):
            self.init_func(num_timesteps, num_sampling_steps)
    
    def test_wrong_timesteps_greater_value(self):
        num_timesteps = 2000
        num_sampling_steps = 30

        with self.assertRaises(ModelInitError):
            self.init_func(num_timesteps, num_sampling_steps)

    def test_wrong_sampling_steps_nagative_value(self):
        num_timesteps = 1000
        num_sampling_steps = -30

        with self.assertRaises(ModelInitError):
            self.init_func(num_timesteps, num_sampling_steps)
    
    def test_wrong_sampling_steps_greater_value(self):
        num_timesteps = 1000
        num_sampling_steps = 600

        with self.assertRaises(ModelInitError):
            self.init_func(num_timesteps, num_sampling_steps)


class TestRFlowScheduler(unittest.TestCase):
    def test_init(self):
        with patch.dict('sys.modules', {'opensora': None}):
            with self.assertRaises(ModelInitError) as context:
                RFlowScheduler(num_timesteps=500, num_sampling_steps=20)
            self.assertIn("Failed to find the opensora library", str(context.exception))


class TestValidateAddNoise(unittest.TestCase):
    @validate_add_noise
    def add_noise(self, original_samples, noise, timesteps):
        return original_samples + noise

    def test_valid_parameters(self):
        original_samples = torch.randn(3, 3, 3, 3, 3)
        noise = torch.randn(3, 3, 3, 3, 3)
        timesteps = torch.randn(5)

        result = self.add_noise(original_samples, noise, timesteps)
        self.assertIsInstance(result, torch.Tensor)

    def test_invalid_original_samples(self):
        original_samples = torch.randn(3, 3, 3, 3, 3).detach().numpy()  # 错误类型
        noise = torch.randn(3, 3, 3, 3, 3).detach().numpy()
        timesteps = torch.randn(5).detach().numpy()

        with self.assertRaises(ParametersInvalid):
            _ = self.add_noise(original_samples, noise, timesteps)
    
    def test_invalid_dim_original_samples(self):
        original_samples = torch.randn(3, 3, 3, 3)  # 错误维度
        noise = torch.randn(3, 3, 3, 3, 3)
        timesteps = torch.randn(5)

        with self.assertRaises(ParametersInvalid):
            _ = self.add_noise(original_samples, noise, timesteps)

    def test_invalid_noise(self):
        original_samples = torch.randn(3, 3, 3, 3, 3).detach().numpy()
        noise = torch.randn(3, 3, 3, 3, 3).detach().numpy()  # 错误类型
        timesteps = torch.randn(5).detach().numpy()

        with self.assertRaises(ParametersInvalid):
            _ = self.add_noise(original_samples, noise, timesteps)
    
    def test_invalid_dim_noise(self):
        original_samples = torch.randn(3, 3, 3, 3, 3)
        noise = torch.randn(3, 3, 3, 3)  # 错误维度
        timesteps = torch.randn(5)

        with self.assertRaises(ParametersInvalid):
            _ = self.add_noise(original_samples, noise, timesteps)

    def test_invalid_timesteps(self):
        original_samples = torch.randn(3, 3, 3, 3, 3).detach().numpy()
        noise = torch.randn(3, 3, 3, 3, 3).detach().numpy()
        timesteps = torch.randn(5).detach().numpy()  # 错误类型

        with self.assertRaises(ParametersInvalid):
            _ = self.add_noise(original_samples, noise, timesteps)
    
    def test_invalid_dim_timesteps(self):
        original_samples = torch.randn(3, 3, 3, 3, 3)
        noise = torch.randn(3, 3, 3, 3, 3)
        timesteps = torch.randn(5, 5)  # 错误维度

        with self.assertRaises(ParametersInvalid):
            _ = self.add_noise(original_samples, noise, timesteps)


if __name__ == '__main__':
    unittest.main()