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
import torch
import safetensors.torch


SAFETENSORS_EXTENSION = "safetensors"
EMA_STATE_DICT = "ema_state_dict"
STATE_DICT = "state_dict"
CPU = "cpu"


def load_state_dict(model_path):
    name = os.path.basename(model_path).split('.')[-1] # get weights name
    if name.endswith("ckpt"):
        weight = torch.load(model_path, map_location=CPU, weights_only=True)
        if (EMA_STATE_DICT in weight):
            weight = weight[EMA_STATE_DICT]
            weight = {key.replace("module.", ""): value for key, value in weight.items()}
        elif STATE_DICT in weight:
            weight = weight[STATE_DICT]
        return weight
    elif name == SAFETENSORS_EXTENSION: # diffuser model use same name
        return safetensors.torch.load_file(model_path, device=CPU) # first load on cpu
    else:
        # to support hf shard model weights
        return torch.load(model_path, map_location=CPU, weights_only=True) # first load on cpu