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

import unittest
import sys
sys.path.append('../')
from mindiesd.schedulers.scheduler_utils import DiffusionScheduler
from mindiesd.utils.logs.logging import logger

CONFIG_NAME = "./configs"
SCHEDULER_CONFIG_NAME = 'scheduler_config_test.json'
SCHEDULER_CONFIG_INVALID_NAME = 'scheduler_config_invalid_test.json'


class Scheduler(DiffusionScheduler):
    config_name = SCHEDULER_CONFIG_NAME

    def __init__(self, num_timesteps, num_sampling_steps, sample_method, loc):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.sample_method = sample_method
        self.loc = loc


class InvalidScheduler(DiffusionScheduler):
    config_name = SCHEDULER_CONFIG_INVALID_NAME

    def __init__(self, num_timesteps, num_sampling_steps, sample_method, loc):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.sample_method = sample_method
        self.loc = loc


class TestDiffusionScheduler(unittest.TestCase):

    def test_from_config(self):
        scheduler = Scheduler.from_config(CONFIG_NAME)
        self.assertEqual(scheduler.num_timesteps, 30)
        self.assertEqual(scheduler.num_sampling_steps, 1000)
        self.assertEqual(scheduler.sample_method, "UNIFORM_CONSTANT")
        self.assertEqual(scheduler.loc, 0.0)

    def test_from_invalid_config(self):
        try:
            invalid_scheduler = InvalidScheduler.from_config(CONFIG_NAME)
        except Exception as e:
            logger.error(e)
            invalid_scheduler = None
        self.assertIsNone(invalid_scheduler)


if __name__ == '__main__':
    unittest.main()