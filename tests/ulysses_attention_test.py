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

import sys
import os
import torch
import torch_npu
import torch.distributed as dist

sys.path.append('../')
from mindiesd.layers.attention import ReconstitutionAttention, SeqParallelAttnProcessor
from mindiesd.utils import logger


def all_gather(tensor: torch.Tensor):
    tensor = tensor.contiguous()
    tensor_list = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=1)


if __name__ == '__main__':

    '''
    Example:
    >>>ASCEND_RT_VISIABLE_DEVICES=2,3 torchrun --master_port=2003 --nproc_per_node=2 ./ulysses_attention_test.py
    '''

    torch.manual_seed(1234) # 1234: rand seed
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    logger.info("rank, local_rank, world_size: %d, %d, %d", rank, local_rank, world_size)
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

    attn = ReconstitutionAttention(
        attention_dim=1152,
        cross_attention_dim=None,
        num_heads=16,
        head_dim=72,
        qkv_bias=True,
        out_proj_bias=True,
        attention_norm='llama_rms_norm',
    )
    device = 'npu'
    attn.to(torch.float16).to(device)

    hidden_states = torch.randn([1, 1024, 1152], dtype=torch.float16).to(device)

    result = attn(hidden_states)

    # 测试并行流程
    attn.set_processor(SeqParallelAttnProcessor())
    sq_paralle_result = attn(hidden_states.split(1024 // world_size, dim=1)[rank])
    sq_paralle_result = all_gather(sq_paralle_result)

    cosine_similarity = torch.cosine_similarity(result.reshape(1, -1), sq_paralle_result.reshape(1, -1))

    logger.info("cosine_similarity is %f", cosine_similarity[0])