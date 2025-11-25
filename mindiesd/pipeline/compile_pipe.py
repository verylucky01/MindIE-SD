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

import torch.nn as nn
from ..utils import is_npu_available


def compile_pipe(pipe, cfg=None):
    if is_npu_available():
        device = 'npu'
        if hasattr(pipe, "text_encoder") and isinstance(pipe.text_encoder, nn.Module):
            pipe.text_encoder.to(device)
        if hasattr(pipe, "transformer") and isinstance(pipe.transformer, nn.Module):
            pipe.transformer.to(device)
        if hasattr(pipe, "vae") and isinstance(pipe.vae, nn.Module):
            pipe.vae.to(device)
        return pipe
    else:
        raise RuntimeError("NPU is not available.")
