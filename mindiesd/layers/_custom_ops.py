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
from typing import Tuple
import torch
from . import register_ops


def rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mode: int
) -> torch.Tensor:
    return getattr(torch.ops.mindie, "rope_mindie_sd")(x, cos, sin, mode)


@register_ops.register_mindie_fake_op("rope_mindie_sd")
def rope_fake(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mode: int
) -> torch.Tensor:
    return torch.empty_like(x)


def laser_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor | None = None,
    alibi_mask: torch.Tensor | None = None,
    drop_mask: torch.Tensor | None = None,
    scale_value: float | torch.Tensor = 1.0,
    head_num: int = 2,
    input_layout: str = "BNSD",
    keep_prob: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 1,
    is_high_precision: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return getattr(torch.ops.mindie, "la_mindie_sd")(
        query=query,
        key=key,
        value=value,
        atten_mask=atten_mask,
        alibi_mask=alibi_mask,
        drop_mask=drop_mask,
        scale_value=scale_value,
        head_num=head_num,
        input_layout=input_layout,
        keep_prob=keep_prob,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        is_highPrecision=is_high_precision,
    )


@register_ops.register_mindie_fake_op("la_mindie_sd")
def attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor = None,
    alibi_mask: torch.Tensor = None,
    drop_mask: torch.Tensor = None,
    scale_value: float = 1.0,
    head_num: int = 2,
    input_layout: str = "BNSD",
    keep_prob: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 1,
    is_high_precision: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    softmax_log_max_sum = torch.empty(
        [query.shape[0], query.shape[1], query.shape[2]],
        device=query.device, dtype=query.dtype
    )
    output = torch.empty_like(query)
    return softmax_log_max_sum, output


def laser_attention_preprocess(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    align_len: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return getattr(torch.ops.mindie, "la_preprocess_mindie_sd")(query, key, value, align_len)


@register_ops.register_mindie_fake_op("la_preprocess_mindie_sd")
def attention_preprocess_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    align_len: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = query.shape[0]
    head_num = query.shape[2] if query.dim() == 4 else query.shape[1]
    head_dim = query.shape[3] if query.dim() == 4 else query.shape[2]
    q_seq_len = query.shape[1] if query.dim() == 4 else query.shape[0]
    k_seq_len = key.shape[1] if key.dim() == 4 else key.shape[0]
    v_seq_len = value.shape[1] if value.dim() == 4 else value.shape[0]
    
    def pad_length(length):
        return (length + align_len - 1) // align_len * align_len
    
    q_padded_seq_len = pad_length(q_seq_len)
    k_padded_seq_len = pad_length(k_seq_len)
    v_padded_seq_len = pad_length(v_seq_len)
    
    def create_padded_tensor(tensor, padded_seq_len):
        if tensor.dim() == 4:
            return torch.empty(
                [batch_size, padded_seq_len, head_num, head_dim],
                device=tensor.device, dtype=tensor.dtype
            )
        else:
            return torch.empty(
                [padded_seq_len, head_num, head_dim],
                device=tensor.device, dtype=tensor.dtype
            )
    
    out_query = create_padded_tensor(query, q_padded_seq_len)
    out_key = create_padded_tensor(key, k_padded_seq_len)
    out_value = create_padded_tensor(value, v_padded_seq_len)
    
    return out_query, out_key, out_value