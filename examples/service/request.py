#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Optional

import torch
import torch_npu
from pydantic import BaseModel


torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class GeneratorRequest(BaseModel):
    prompt: str
    sample_steps: int
    base_seed: Optional[int] = 0
    save_disk_path: Optional[str] = None
    size: Optional[str] = '1280*720'
    sample_guide_scale: Optional[float] = None
    frame_num: Optional[int] = 81
    sample_shift: Optional[float] = None
    sample_solver: Optional[str] = 'unipc'
    offload_model: Optional[bool] = False
    sample_fps: Optional[int] = 16
    task: Optional[str] = 't2v-A14B'
    ckpt_dir: Optional[str] = '/data'
    image: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "a beautiful landscape",
                "sample_steps": 40,
                "base_seed": 0,
                "save_disk_path": "/home/",
                "size": "1280*720"
            }
        }