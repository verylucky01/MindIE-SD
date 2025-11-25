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
import inspect
import importlib

import torch
from tqdm import tqdm

from ..config_utils import ConfigMixin
from ..utils import file_utils


PIPELINE_CONFIG_NAME = "model_index.json"
OPEN_SORA_DEFAULT_VIDEO_FRAME = 32
OPEN_SORA_DEFAULT_IMAGE_SIZE = (720, 1280)
OPEN_SORA_DEFAULT_FPS = 8
ENABLE_SEQUENCE_PARALLELISM_DEFAULT_VALUE = False

VAE = 'vae'
TEXT_ENCODER = 'text_encoder'
TOKENIZER = 'tokenizer'
TRANSFORMER = 'transformer'
SCHEDULER = 'scheduler'
NUM_FRAMES = 'num_frames'
IMAGE_SIZE = 'image_size'
ENABLE_SEQUENCE_PARALLELISM = 'enable_sequence_parallelism'
FPS = 'fps'
DTYPE = 'dtype'
SET_PATCH_PARALLEL = 'set_patch_parallel'
FROM_PRETRAINED = 'from_pretrained'

key_default_values = {NUM_FRAMES: OPEN_SORA_DEFAULT_VIDEO_FRAME, IMAGE_SIZE: OPEN_SORA_DEFAULT_IMAGE_SIZE,
                      FPS: OPEN_SORA_DEFAULT_FPS,
                      ENABLE_SEQUENCE_PARALLELISM: ENABLE_SEQUENCE_PARALLELISM_DEFAULT_VALUE,
                      SET_PATCH_PARALLEL: False, "dtype": torch.bfloat16}


class OpenSoraPipeline(ConfigMixin):
    r"""
    Base class for all OpenSora pipelines.
    The OpenSoraPipeline class is mainly provides `from_pretrained` method to
    initialize the OpenSora pipeline components from `config_name` file,
    and loads the weights of the components from `model_path` directory.
    """

    config_name = PIPELINE_CONFIG_NAME

    def __init__(self):
        super().__init__()

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        r"""
        The from_pretrained class method is used to initialize the OpenSora pipeline components
        and loads the weights of the components from the model directory.

        Args:
            model_path (str): The path to the model directory.
            **kwargs: Additional keyword arguments for the pipeline components.

        Returns:
            OpenSoraPipeline: The initialized OpenSora pipeline.
        """
        [num_frames, image_size, fps, enable_sequence_parallelism, set_patch_parallel, dtype] =\
            _get_values_from_kwargs(kwargs, key_default_values)
        model_path = file_utils.standardize_path(model_path)
        file_utils.check_dir_safety(model_path, permission_mode=file_utils.MODELDATA_DIR_PERMISSION)
        init_dict, _ = cls.load_config(model_path, **kwargs)

        init_list = [VAE, TEXT_ENCODER, TOKENIZER, TRANSFORMER, SCHEDULER]
        pipe_init_dict = {}

        all_parameters = inspect.signature(cls.__init__).parameters

        required_param = {k: v for k, v in all_parameters.items() if v.default == inspect.Parameter.empty}
        expected_modules = set(required_param.keys()) - {"self"}
        # init the module from kwargs
        passed_module = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}

        for key in tqdm(init_list, desc="Loading open-sora-pipeline compenents"):
            if key not in init_dict:
                raise ValueError(f"Failed to obtain the key:{key} from the initial configuration!")
            if key in passed_module:
                pipe_init_dict[key] = passed_module.pop(key)
            else:
                modules, cls_name = init_dict[key]
                if modules == "mindiesd":
                    library = importlib.import_module("opensora")
                else:
                    library = importlib.import_module(modules)
                class_obj = getattr(library, cls_name)

                sub_folder = os.path.join(model_path, key)

                from_pretrained = kwargs.pop(FROM_PRETRAINED, sub_folder)
                from_pretrained = file_utils.standardize_path(from_pretrained)
                file_utils.check_dir_safety(from_pretrained, permission_mode=file_utils.MODELDATA_DIR_PERMISSION)

                if key == TRANSFORMER:
                    _check(pipe_init_dict)
                    latent_size = pipe_init_dict.get(VAE).get_latent_size((num_frames, *image_size))
                    in_channels = pipe_init_dict.get(VAE).out_channels
                    caption_channels = pipe_init_dict.get(TEXT_ENCODER).config.d_model
                    pipe_init_dict[key] = class_obj.from_pretrained(sub_folder, input_size=latent_size,
                        in_channels=in_channels, caption_channels=caption_channels,
                        enable_sequence_parallelism=enable_sequence_parallelism, dtype=dtype,
                        trust_remote_code=False, **kwargs)
                else:
                    initializer = _get_initializers().get(key, init_default)
                    pipe_init_dict[key] = initializer(class_obj, sub_folder, from_pretrained,
                                                      set_patch_parallel, dtype=dtype, **kwargs)

        pipe_init_dict[NUM_FRAMES] = num_frames
        pipe_init_dict[IMAGE_SIZE] = image_size
        pipe_init_dict[FPS] = fps
        pipe_init_dict[DTYPE] = dtype

        return cls(**pipe_init_dict)


def _get_values_from_kwargs(kwargs, key_values: dict):
    ret_key_values = []
    for key, value in key_values.items():
        ret_key_values.append(kwargs.pop(key, value))
    return ret_key_values


def _get_initializers():
    initializers = {
        TEXT_ENCODER: init_text_encoder, SCHEDULER: init_scheduler,
        VAE: init_vae, TOKENIZER: init_default
    }
    return initializers


def _check(pipe_init_dict):
    if pipe_init_dict.get(VAE) is None:
        raise ValueError("Failed to get the module 'vae' from the initialization list!")
    if not hasattr(pipe_init_dict.get(VAE), 'get_latent_size'):
        raise ValueError("Vae has no attribute 'get_latent_size'!")
    if pipe_init_dict.get(TEXT_ENCODER) is None:
        raise ValueError("Failed to get the module 'text_encoder' from the initialization list!")


def init_text_encoder(class_obj, sub_folder, from_pretrained, set_patch_parallel, **kwargs):
    file_utils.check_file_under_dir(sub_folder)
    return class_obj.from_pretrained(sub_folder,
                                     local_files_only=True,
                                     trust_remote_code=False).to(kwargs.get(DTYPE))


def init_scheduler(class_obj, sub_folder, from_pretrained, set_patch_parallel, **kwargs):
    return class_obj.from_config(sub_folder, trust_remote_code=False)


def init_vae(class_obj, sub_folder, from_pretrained, set_patch_parallel, **kwargs):
    vae = class_obj.from_pretrained(sub_folder,
                                    from_pretrained=from_pretrained,
                                    set_patch_parallel=set_patch_parallel,
                                    trust_remote_code=False,
                                    **kwargs)
    vae.to(kwargs.get(DTYPE))
    return vae


def init_default(class_obj, sub_folder, from_pretrained, set_patch_parallel, **kwargs):
    file_utils.check_file_under_dir(sub_folder)
    return class_obj.from_pretrained(sub_folder,
                                     local_files_only=True,
                                     trust_remote_code=False,
                                     **kwargs)
