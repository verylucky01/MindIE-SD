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

import functools
from typing import Tuple
import torch

from mindiesd.config_utils import ConfigMixin
from mindiesd.models.model_utils import DiffusionModel
from mindiesd.utils import ConfigError, ParametersInvalid, ModelInitError


MAX_IN_CHANNELS = 4
MAX_CAPTIOIN_CHANNELS = 4096


def check_init_params(func):
    @functools.wraps(func)
    def wrapper(self, config):
        # Define expected parameter types and names
        expected_params = {
            'input_size': (tuple, list),
            'in_channels': int,
            'caption_channels': int,
            'enable_flash_attn': bool,
            'enable_sequence_parallelism': bool,
            'use_cache': bool,
            'cache_interval': int,
            'cache_start': int,
            'num_cache_layer': int,
            'cache_start_steps': int
        }

        # 检查STDiT3Config中的值是否符合类型，不符合发挥TypeError
        for param_name, param_type in expected_params.items():
            if not isinstance(getattr(config, param_name), param_type):
                raise ConfigError(
                    f'The data type of the parameter:{param_name} must be:{param_type}, '
                    f'but got {type(getattr(config, param_name))}.'
                )
        
        # Define expected parameter targets
        expected_targets = {
            'in_channels': MAX_IN_CHANNELS,
            'caption_channels': MAX_CAPTIOIN_CHANNELS
        }
        
        # 检查STDiT3Config中的值是否都在目标范围
        for param_name, param_target in expected_targets.items():
            if getattr(config, param_name) < 1 or getattr(config, param_name) > param_target:
                raise ConfigError(
                    f"The value of the parameter:{param_name} must be in {[1, param_target]}, "
                    f"but got {getattr(config, param_name)}."
                )

        return func(self, config)

    return wrapper


def check_forward_params(func):
    @functools.wraps(func)
    def wrapper(self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, t_idx=0, **kwargs):
        param_types = {
            "x": torch.Tensor,
            "timestep": torch.Tensor,
            "y": torch.Tensor,
            "mask": (torch.Tensor, type(None)),
            "x_mask": (torch.Tensor, type(None)),
            "fps": (torch.Tensor, type(None)),
            "height": (torch.Tensor, type(None)),
            "width": (torch.Tensor, type(None)),
            "t_idx": int
        }

        for param_name, expected_type in param_types.items():
            param_value = locals()[param_name]
            if not isinstance(param_value, expected_type):
                raise ParametersInvalid(f"The data type of the parameter:{param_name} must be {expected_type}, "
                                        f"but got {type(param_value)}.")

        if x.dim() != 5:
            raise ParametersInvalid(f"x must be a 5D tensor, but got {x.dim()}D.")
        if timestep.dim() != 1:
            raise ParametersInvalid(f"The timestep must be a 1D tensor, but got {timestep.dim()}D.")
        if y.dim() != 4:
            raise ParametersInvalid(f"y must be a 4D tensor, but got {y.dim()}D.")
        if mask is not None and mask.dim() != 2:
            raise ParametersInvalid(f"mask must be a 2D tensor or None, but got {mask.dim()}D.")
        if x_mask is not None and x_mask.dim() != 2:
            raise ParametersInvalid(f"x_mask must be a 2D tensor or None, but got {x_mask.dim()}D.")
        if height is not None and height.dim() != 1:
            raise ParametersInvalid(f"height must be a 1D tensor or None, but got {height.dim()}D.")
        if width is not None and width.dim() != 1:
            raise ParametersInvalid(f"width must be a 1D tensor or None, but got {width.dim()}D.")

        return func(self, x, timestep, y, mask, x_mask, fps, height, width, t_idx, **kwargs)

    return wrapper


class STDiT3Config(ConfigMixin):
    config_name = 'config.json'

    def __init__(
            self,
            input_size: Tuple[int, int, int] = (None, None, None),
            in_channels: int = 4,
            caption_channels: int = 4096,
            enable_flash_attn: bool = True,
            enable_sequence_parallelism: bool = False,
            use_cache: bool = True,
            cache_interval: int = 2,
            cache_start: int = 3,
            num_cache_layer: int = 13,
            cache_start_steps: int = 5,
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.caption_channels = caption_channels
        self.enable_flash_attn = enable_flash_attn
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.use_cache = use_cache
        self.cache_interval = cache_interval
        self.cache_start = cache_start
        self.num_cache_layer = num_cache_layer
        self.cache_start_steps = cache_start_steps 


class STDiT3(DiffusionModel):
    config_class = STDiT3Config
    weights_name = 'model.safetensors'

    @check_init_params
    def __init__(self, config):
        super().__init__(config)
        try:
            from opensora import STDiT3 as STDiT3Model
        except Exception as e:
            raise ModelInitError("Failed to find the opensora library. Please set the environment path \
                                 by referring to the mindiesd development manual.") from e
        self.stdit3_model = STDiT3Model(config)

    @check_forward_params
    def forward(
        self, 
        x: torch.Tensor, 
        timestep: torch.Tensor, 
        y: torch.Tensor, 
        mask: torch.Tensor = None, 
        x_mask: torch.Tensor = None, 
        fps: torch.Tensor = None, 
        height: torch.Tensor = None, 
        width: torch.Tensor = None, 
        t_idx: int = 0, 
        **kwargs
    ) -> torch.Tensor:
        return self.stdit3_model(x, timestep, y, mask, x_mask,
                                 fps, height, width, t_idx)
