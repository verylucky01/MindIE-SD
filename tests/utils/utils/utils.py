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

import random
import torch
import numpy as np

IMG_FPS = 8


def set_random_seed(seed):
    """Set random seed.
    Args:
        seed (int, optional): Seed to be used.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def prepare_multi_resolution_info(info_type, video_property, device, dtype):
    (batch_size, image_size, num_frames, fps) = video_property
    if info_type is None:
        return dict()
    elif info_type == "PixArtMS":
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(batch_size, 1)
        return dict(ar=ar, hw=hw)
    elif info_type in ["STDiT2", "OpenSora"]:
        fps = fps if num_frames > 1 else IMG_FPS
        fps = torch.tensor([fps], device=device, dtype=dtype).repeat(batch_size)
        height = torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(batch_size)
        width = torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(batch_size)
        num_frames = torch.tensor([num_frames], device=device, dtype=dtype).repeat(batch_size)
        ar = torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype).repeat(batch_size)
        return dict(height=height, width=width, num_frames=num_frames, ar=ar, fps=fps)
    else:
        raise NotImplementedError


def extract_prompts_loop(prompts, num_loop=0):
    ret_prompts = []
    for prompt in prompts:
        if prompt.startswith("|0|"):
            prompt_list = prompt.split("|")[1:]
            text_list = []
            for i in range(0, len(prompt_list), 2):
                start_loop = int(prompt_list[i])
                text = prompt_list[i + 1]
                end_loop = int(prompt_list[i + 2]) if i + 2 < len(prompt_list) else num_loop + 1
                text_list.extend([text] * (end_loop - start_loop))
            prompt = text_list[num_loop]
        ret_prompts.append(prompt)
    return ret_prompts


def split_prompt(prompt_text):
    if prompt_text.startswith("|0|"):
        # this is for prompts which look like
        # |0| a beautiful day |1| a sunny day |2| a rainy day
        # we want to parse it into a list of prompts with the loop index
        prompt_list = prompt_text.split("|")[1:]
        text_list = []
        loop_idx = []
        for i in range(0, len(prompt_list), 2):
            start_loop = int(prompt_list[i])
            text = prompt_list[i + 1].strip()
            text_list.append(text)
            loop_idx.append(start_loop)
        return text_list, loop_idx
    else:
        return_value = None
        return [prompt_text], return_value


def merge_prompt(text_list, loop_idx_list=None):
    if loop_idx_list is None:
        return text_list[0]
    else:
        prompt = ""
        for i, text in enumerate(text_list):
            prompt += f"|{loop_idx_list[i]}|{text}"
        return prompt


def append_score_to_prompts(prompts, aes=None, flow=None, camera_motion=None):
    new_prompts = []
    for prompt in prompts:
        new_prompt = prompt
        if aes is not None and "aesthetic score:" not in prompt:
            new_prompt = f"{new_prompt} aesthetic score: {aes:.1f}."
        if flow is not None and "motion score:" not in prompt:
            new_prompt = f"{new_prompt} motion score: {flow:.1f}."
        if camera_motion is not None and "camera motion:" not in prompt:
            new_prompt = f"{new_prompt} camera motion: {camera_motion}."
        new_prompts.append(new_prompt)
    return new_prompts
