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

from functools import wraps
from typing import List

import torch
from mindiesd.schedulers.scheduler_utils import DiffusionScheduler
from mindiesd.utils import ModelInitError, ParametersInvalid


MAX_NUM_TIMESTEPS = 1000
MAX_NUM_SAMPLING_STEPS = 500


def validate_add_noise(func):
    @wraps(func)
    def wrapper(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    ):
        # 检查入参
        if not isinstance(original_samples, torch.Tensor):
            raise (ParametersInvalid(
                f"The data type of the input original_samples must be torch.Tensor, "
                f"but got {type(original_samples)}."))
        if not isinstance(noise, torch.Tensor):
            raise ParametersInvalid(f"The data type of the input noise must be torch.Tensor, but got {type(noise)}.")
        if not isinstance(timesteps, torch.Tensor):
            raise ParametersInvalid(f"The data type of the input timesteps must be torch.Tensor, "
                                    f"but got {type(timesteps)}.")
        if original_samples.dim() != 5:
            raise ParametersInvalid(f"The input original_samples must be a 5D tensor, "
                                    f"but got {original_samples.dim()}D.")
        if noise.dim() != 5:
            raise ParametersInvalid(f"The input noise must be a 5D tensor, but got {noise.dim()}D.")
        if timesteps.dim() != 1:
            raise ParametersInvalid(f"The input timesteps must be a 1D tensor, but got {timesteps.dim()}D.")
        return func(self, original_samples, noise, timesteps)

    return wrapper


def validate_step(func):
    @wraps(func)
    def wrapper(
            self, pred: torch.Tensor,
            timesteps: List[float],
            i: int,
            noise: torch.Tensor,
            guidance_scale: float = 7.0
    ):
        # 检查入参
        if not isinstance(pred, torch.Tensor):
            raise ParametersInvalid(f'The data type of the input pred must be torch.Tensor, but got {type(pred)}).')
        if not isinstance(timesteps, List):
            raise ParametersInvalid(f'The data type of the input timesteps must be List, but got {type(timesteps)}.')
        if not isinstance(i, int):
            raise ParametersInvalid(f'The data type of the input i must be int, but got {type(i)}.')
        if not isinstance(noise, torch.Tensor):
            raise ParametersInvalid(f'The data type of the input noise must be torch.Tensor, but got {type(noise)}.')
        if not isinstance(guidance_scale, float):
            raise ParametersInvalid(f'The data type of the input guidance_scale must be float, '
                                    f'but got {type(guidance_scale)}.')
        if pred.dim() != 5:
            raise ParametersInvalid(f"The input pred must be a 5D tensor, but got {pred.dim()}D.")
        if len(timesteps) != self.num_sampling_steps:
            raise ParametersInvalid(f"The length of timesteps must be equal to {self.num_sampling_steps}, "
                                    f"but got {len(timesteps)}.")
        if i < 0 or i >= self.num_sampling_steps:
            raise ModelInitError(f"Input i: {i} must be in range [0, {self.num_sampling_steps}).")
        if noise.dim() != 5:
            raise ParametersInvalid(f"The input noise must be a 5D tensor, but got {noise.dim()}D.")
        return func(self, pred, timesteps, i, noise, guidance_scale)

    return wrapper


def validate_init_params(func):
    @wraps(func)
    def wrapper(
            self,
            num_timesteps: int,
            num_sampling_steps: int,
    ):
        if not isinstance(num_timesteps, int):
            raise ModelInitError(f"The data type of the input num_timesteps must be int, "
                                 f"but got {type(num_timesteps)}.")
        if not isinstance(num_sampling_steps, int):
            raise ModelInitError(f"The data type of the input num_sampling_steps must be int, "
                                 f"but got {type(num_sampling_steps)}.")
        if num_timesteps <= 0 or num_timesteps > MAX_NUM_TIMESTEPS:
            raise ModelInitError(f"Input num_timesteps:{num_timesteps}"
                                 f"must be in range (0, {MAX_NUM_TIMESTEPS}].")
        if num_sampling_steps <= 0 or num_sampling_steps > MAX_NUM_SAMPLING_STEPS:
            raise ModelInitError(f"Input num_sampling_steps:{num_sampling_steps}"
                                 f"must be in range (0, {MAX_NUM_SAMPLING_STEPS}].")
        return func(self, num_timesteps, num_sampling_steps)

    return wrapper


class RFlowScheduler(DiffusionScheduler):
    r""" 
    The scheduler for the Opensora1.2, which is used for sampling , denoising and managing step.
    Inherited from the DiffusionScheduler class.
       
    Args:
        num_timesteps (int): The number of timesteps. The supported range is (0, 1000].
        num_sampling_steps (int): The number of sampling steps. The supported range is (0, 500].
    """

    @validate_init_params
    def __init__(
            self,
            num_timesteps: int = 1000,
            num_sampling_steps: int = 30,
    ):
        super().__init__()
        self.num_sampling_steps = num_sampling_steps
        try:
            from opensora import RFlowScheduler as RFScheduler
        except Exception as e:
            raise ModelInitError("Failed to find the opensora library. Please set the environment path \
                                 by referring to the mindiesd development manual.") from e
        self.rflow_scheduler = RFScheduler(num_timesteps, num_sampling_steps)

    @validate_add_noise
    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Adds noise to the original samples based on the given noise and timesteps.
        compatible with diffusers add_noise()

        Args:
            original_samples (torch.Tensor): The original samples to which noise will be added.
            noise (torch.Tensor): The noise to be added to the original samples.
            timesteps (torch.Tensor): The timesteps at which the noise will be added.

        Returns:
            torch.Tensor: The original samples with noise added.
        """
        return self.rflow_scheduler.add_noise(original_samples, noise, timesteps)

    @validate_step
    def step(self, pred: torch.Tensor,
             timesteps: List[float],
             i: int,
             noise: torch.Tensor,
             guidance_scale: float = 7.0) -> torch.Tensor:
        r"""
        Updates the noise based on the given prediction, timesteps, noise, and guidance scale.

        Args:
            pred (torch.Tensor): The prediction used to update the noise.
            timesteps (List[float]): The list contains all the timesteps .
            i (int): The index of the current timestep.
            noise (torch.Tensor): The noise to be updated.
            guidance_scale (float): The guidance scale used to update the noise on the
                conditional predition and unconditional predition. Default is 7.0.

        Returns:
            torch.Tensor: The updated noise.
        """
        return self.rflow_scheduler.step(pred, timesteps, i, noise, guidance_scale)