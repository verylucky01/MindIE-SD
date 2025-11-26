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

from typing import Tuple, List

import torch
import torch_npu

from transformers import T5TokenizerFast, T5EncoderModel

from .pipeline_utils import OpenSoraPipeline
from ..models import STDiT3, VideoAutoencoder
from ..schedulers import RFlowScheduler
from ..utils import ParametersInvalid, ModelInitError

torch_npu.npu.config.allow_internal_format = False


target_image_size = [(720, 1280), (512, 512)]
target_num_frames = [32, 128]
target_fps = [8]
target_output_type = ["latent", "thwc"]
target_dtype = [torch.bfloat16, torch.float16]
MAX_PROMPT_LENGTH = 1024 # the limits of open-sora1.2


def check_init_input(input_models, input_params):
    text_encoder, tokenizer, transformer, vae, scheduler = input_models
    num_frames, image_size, fps, dtype = input_params
    if text_encoder.__class__.__name__ != "T5EncoderModel":
        raise ParametersInvalid(f"The data type of input text_encoder must be \
                                T5EncoderModel, but got {type(text_encoder)}.")
    if tokenizer.__class__.__name__ != "T5TokenizerFast":
        raise ParametersInvalid(f"The data type of input tokenizer must be T5TokenizerFast, but got {type(tokenizer)}.")
    if transformer.__class__.__name__ != "STDiT3":
        raise ParametersInvalid(f"The data type of input transformer must be STDiT3, but got {type(transformer)}.")
    if vae.__class__.__name__ != "VideoAutoencoder":
        raise ParametersInvalid(f"The data type of input vae must be VideoAutoencoder, but got {type(vae)}.")
    if scheduler.__class__.__name__ != "RFlowScheduler":
        raise ParametersInvalid(f"The data type of input scheduler must be RFlowScheduler, but got {type(scheduler)}.")
    if not isinstance(num_frames, int):
        raise ParametersInvalid(f"The data type of input num_frames must be int, but got {type(num_frames)}.")
    if not isinstance(image_size, tuple):
        raise ParametersInvalid(f"The data type of input image_size must be tuple, but got {type(image_size)}.")
    if not isinstance(fps, int):
        raise ParametersInvalid(f"The data type of input fps must be int, but got {type(fps)}.")
    if not isinstance(dtype, torch.dtype):
        raise ParametersInvalid(f"The data type of input dtype must be torch.dtype, but got {type(dtype)}.")

    value_checks = {
        "num_frames": (num_frames, target_num_frames),
        "image_size": (image_size, target_image_size),
        "fps": (fps, target_fps),
        "dtype": (dtype, target_dtype)
    }
    for attr, (val, target_vals) in value_checks.items():
        if val not in target_vals:
            raise ModelInitError(f"{attr} must be expected target of {target_vals}")


def check_call_input(output_type, prompts, seed):
    if not (isinstance(prompts, list) and all(isinstance(item, str) for item in prompts)):
        raise ParametersInvalid("The data type of the input prompts must be a list of strings.")
    if not isinstance(seed, int):
        raise ParametersInvalid("The data type of the input seed must be int.")
    if not isinstance(output_type, str):
        raise ParametersInvalid("The data type of the input output_type must be string.")
    if len(prompts) != 1:
        raise ParametersInvalid(f"The length of prompts must be 1, but got {len(prompts)}.")
    if len(prompts[0]) == 0 or len(prompts[0]) >= MAX_PROMPT_LENGTH:
        raise ParametersInvalid(
            f"The length of the 0th prompt should be in range (0, {MAX_PROMPT_LENGTH}), but got {len(prompts[0])}.")
    if seed < 0:
        raise ParametersInvalid(f"Input seed should be a non-negative integer, but got {seed}.")
    if output_type not in target_output_type:
        raise ParametersInvalid(f"The output_type:{output_type} not in target_output_type:{target_output_type}.")


class OpenSoraPipeline12(OpenSoraPipeline):

    def __init__(self, text_encoder: T5EncoderModel, tokenizer: T5TokenizerFast, transformer: STDiT3,
                 vae: VideoAutoencoder, scheduler: RFlowScheduler,
                 num_frames: int = 32, image_size: Tuple[int, int] = (720, 1280), fps: int = 8,
                 dtype: torch.dtype = torch.bfloat16):

        super().__init__()
        try:
            from opensora import OpenSoraPipeline12 as opensora12
        except Exception as e:
            raise ModelInitError("Failed to find the opensora library. Please set the environment path \
                                 by referring to the mindiesd development manual.") from e

        input_models = (text_encoder, tokenizer, transformer, vae, scheduler)
        input_params = (num_frames, image_size, fps, dtype)
        check_init_input(input_models, input_params)

        self.opensora_pipeline = opensora12(text_encoder, tokenizer, transformer,
                                            vae, scheduler, num_frames, image_size, fps, dtype)

    @torch.no_grad()
    def __call__(self, prompts: List[str], seed: int = 42, output_type: str = "latent"):
        check_call_input(output_type, prompts, seed)
        return self.opensora_pipeline(prompts, seed, output_type) 