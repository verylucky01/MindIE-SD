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
import json

import torch
import torch.nn as nn

from .model_load_utils import load_state_dict
from ..config_utils import ConfigMixin
from ..utils import file_utils
from ..utils import ModelInitError

DIFFUSER_SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"


class DiffusionModel(nn.Module):
    config_class = ConfigMixin
    weights_name = DIFFUSER_SAFETENSORS_WEIGHTS_NAME

    def __init__(self, config):
        super().__init__()
        self.config = config

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        dtype = kwargs.pop('dtype', None) # get dtype from kwargs
        model_path = file_utils.standardize_path(model_path)
        file_utils.check_dir_safety(model_path, permission_mode=file_utils.MODELDATA_DIR_PERMISSION)
        if not issubclass(cls.config_class, ConfigMixin):
            raise ModelInitError("config_class is not a subclass of ConfigMixin.")
        weights_path = os.path.join(model_path, cls.weights_name)

        is_sharded = weights_path.endswith(".index.json")
        # 1. check weights_path and weight file

        if not is_sharded:
            weights_path = weights_path
            weights_path = file_utils.standardize_path(weights_path)
            file_utils.check_file_safety(weights_path)

        else:
            index_filename = weights_path
            with open(index_filename) as f:
                index = json.loads(f.read())
            if "weight_map" in index:
                index = index["weight_map"]
            if isinstance(index, dict):
                weights_path = sorted(list(set(index.values())))
            else:
                raise ValueError(
                    f"The json config object must be dict, but got: {type(index)}"
                )
            weights_path = [os.path.join(model_path, f) for f in weights_path]
            for i, weight_file in enumerate(weights_path.copy()):
                weight_file = file_utils.standardize_path(weight_file)
                file_utils.check_file_safety(weight_file)
                weights_path[i] = weight_file

        # 2. load config_class from json
        init_dict, _ = cls.config_class.load_config(model_path, **kwargs)
        config = cls.config_class(**init_dict)

        # 3. init model with config
        model = cls(config)

        # 4. load model weights
        model = cls._load_model(model, weights_path, is_sharded)
        # 5. model to dtype
        if dtype is not None:
            model.to(dtype)
        return model

    @classmethod
    def _load_model(cls, model, weights_path, is_sharded):
        if not is_sharded:
            state_dict = load_state_dict(weights_path)
            model.load_weights(state_dict)
        else:
            need_key = set(model.state_dict().keys())
            state_dict = {}
            cache = {}
            for weight_file in weights_path:
                state_dict = load_state_dict(weight_file)
                state_dict.update(cache)
                loadkey_cache = model.load_weights(state_dict)
                if loadkey_cache:
                    if isinstance(loadkey_cache, tuple):
                        loaded_keys, cache = loadkey_cache
                    else:
                        loaded_keys = loadkey_cache
                need_key = need_key.symmetric_difference(set(loaded_keys))

            if len(need_key) > 0:
                raise ModelInitError(f"The weight misses key: {need_key}.")
        return model
