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
from ..config_utils import ConfigMixin

SCHEDULER_CONFIG_NAME = "scheduler_config.json"

logger = logging.getLogger(__name__)  # init python log


class DiffusionScheduler(ConfigMixin):
    r"""
    Base class for all scheduler.
    The `DiffusionScheduler` class  mainly provides `from_config` method to 
    load configuration and initialize the scheduler.

    Args:
        model_path (str): The path to the scehdule config file.
        **kwargs: Additional keyword arguments for the scheduler.
    """
    config_name = SCHEDULER_CONFIG_NAME

    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, model_path, **kwargs):
        r"""
        The method is used to load the configuration and initialize the scheduler.

        Args:
            model_path (str): The path to the scheduler configuration file.
            **kwargs: Additional keyword arguments for the scheduler.

        Returns:
            DiffusionScheduler: The initialized scheduler.
        """
        init_dict, config_dict = cls.load_config(model_path, **kwargs)
        return cls(**init_dict, **kwargs)