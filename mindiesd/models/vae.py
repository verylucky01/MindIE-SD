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

import os
from functools import wraps
import torch

from mindiesd.config_utils import ConfigMixin
from mindiesd.models.model_utils import DiffusionModel
from mindiesd.utils import file_utils, ParametersInvalid, ConfigError, ModelInitError

VAE_TARGET_NUM_FRAMES = [32, 128]
VAE_TARGET_IMAGE_SIZE = [(720, 1280), (512, 512)]


def check_init_params(func):
    @wraps(func)
    def wrapper(self, config):
        expected_params = {
            'vae_2d': dict,
            'freeze_vae_2d': bool,
            'micro_frame_size': int,
            'shift': tuple,
            'scale': tuple,
            'set_patch_parallel': bool
        }
        if not isinstance(config, VideoAutoencoderConfig):
            raise ConfigError(f"Input config must be {VideoAutoencoderConfig.__name__}")
        for param_name, param_type in expected_params.items():
            if not isinstance(getattr(config, param_name), param_type):
                raise ConfigError(f"The data type of the initial parameter:{param_name} in the config must be"
                                  f"{param_type}, but got {type(getattr(config, param_name))}.")
        return func(self, config)

    return wrapper


def check_get_latent_size(func):
    @wraps(func)
    def wrapper(self, input_size: tuple):
        if not isinstance(input_size, tuple):
            raise ParametersInvalid(f"The data type of input_size must be tuple, but got {type(input_size)}.")
        if len(input_size) != 3:
            raise ParametersInvalid(f"The length of input_size must be 3, but got {len(input_size)}.")
        if input_size[0] not in VAE_TARGET_NUM_FRAMES:
            raise ParametersInvalid(f"The value of input_size[0] must be in {VAE_TARGET_NUM_FRAMES}, "
                                    f"but got {input_size[0]}.")
        if (input_size[1], input_size[2]) not in VAE_TARGET_IMAGE_SIZE:
            raise ParametersInvalid(
                f"The value of input_size[1:] must be in {VAE_TARGET_IMAGE_SIZE}, "
                f"but got {(input_size[1], input_size[2])}.")
        return func(self, input_size)

    return wrapper


def check_decode(func):
    @wraps(func)
    def wrapper(self, z: torch.Tensor, num_frames: int):
        if not isinstance(z, torch.Tensor):
            raise ParametersInvalid(f"The data type of z must be torch.Tensor, but got {type(z)}.")
        if not isinstance(num_frames, int):
            raise ParametersInvalid(f"The data type of num_frames must be int, but got {type(num_frames)}.")
        if z.dim() != 5:
            raise ParametersInvalid(f"The dimension of input z must be 5, but got {z.ndim}")
        if num_frames not in VAE_TARGET_NUM_FRAMES:
            raise ParametersInvalid(f"The value of input num_frames must be in {VAE_TARGET_NUM_FRAMES}, "
                                    f"but got {num_frames}.")
        return func(self, z, num_frames)

    return wrapper


class VideoAutoencoderConfig(ConfigMixin):
    config_name = 'config.json'

    def __init__(
            self,
            from_pretrained,
            set_patch_parallel=False,
            **kwargs,
    ):
        file_utils.check_path_is_none(from_pretrained)
        from_pretrained = os.path.join(from_pretrained, "vae_2d")
        model_path = file_utils.standardize_path(from_pretrained)
        file_utils.check_dir_safety(model_path, permission_mode=file_utils.MODELDATA_DIR_PERMISSION)
        vae_2d = dict(from_pretrained=model_path,
                      subfolder="vae",
                      micro_batch_size=4)
        self.vae_2d = vae_2d
        self.freeze_vae_2d = False
        self.micro_frame_size = 17

        self.shift = (-0.10, 0.34, 0.27, 0.98)
        self.scale = (3.85, 2.32, 2.33, 3.06)

        self.set_patch_parallel = set_patch_parallel

        super().__init__(**kwargs)


class VideoAutoencoder(DiffusionModel):
    config_class = VideoAutoencoderConfig

    weights_name = 'model.safetensors'

    @check_init_params
    def __init__(self, config: VideoAutoencoderConfig):
        super().__init__(config=config)
        try:
            from opensora import VideoAutoencoder as VideoAutoencoderModel
        except Exception as e:
            raise ModelInitError("Failed to find the opensora library. Please set the environment path \
                                 by referring to the mindiesd development manual.") from e
        self.vae_model = VideoAutoencoderModel(config)

    @check_get_latent_size
    def get_latent_size(self, input_size):
        return self.vae_model.get_latent_size(input_size)

    @check_decode
    def decode(self, z, num_frames):
        return self.vae_model.decode(z, num_frames)