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

import os
from pathlib import Path

import torch
import torch_npu
from .attention_operate import AttentionOperateBase, register_op_800
from ...utils.exception import ParametersInvalid
from ...utils import file_utils
from .common import AttentionParam

current_path = Path(__file__).resolve()
if len(current_path.parents) < 3:
    raise ParametersInvalid("The parents level is insufficient.")
ops_path = current_path.parents[2] / "plugin"
ops_path = file_utils.standardize_path(str(ops_path))
ops_file = os.path.join(ops_path, "libPTAExtensionOPS.so")
file_utils.check_file_safety(ops_file, permission_mode=file_utils.MODELDATA_FILE_PERMISSION)
torch.ops.load_library(ops_file)


@register_op_800("ascend_laser_preprocess")
class AscendLaserPreprocess(AttentionOperateBase):
    supported_layout = ["BSND"]
    supported_dtype = [torch.float16, torch.bfloat16]

    @classmethod
    def forward_preprocess(
            cls,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            align_len: int = 256
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):

        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ParametersInvalid("LA_preprocess输入必须是4D张量")
        batch_size, seq_len, head_num, head_dim = query.shape
        original_dtype = query.dtype
        
        attn_param = AttentionParam(
            batch_size=batch_size,
            head_num=head_num,
            q_seqlen=seq_len,
            kv_seqlen=key.shape[1],
            head_dim=head_dim,
            dtype=original_dtype
        )

        out_query, out_key, out_value = torch.ops.mindie.la_preprocess_mindie_sd(
            query, key, value, align_len
        )
        return out_query, out_key, out_value


def la_preprocess(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, align_len: int = 256):
    return AscendLaserPreprocess.forward_preprocess(query, key, value, align_len)
