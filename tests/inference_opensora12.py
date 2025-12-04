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
import argparse
import time
import logging
import torch
from torchvision.io import write_video

from mindiesd import OpenSoraPipeline12, compile_pipe
from tests.utils.utils.parallel_mgr import init_parallel_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=0, help="NPU device id")
    parser.add_argument("--device", type=str, default='npu', help="NPU")
    parser.add_argument("--type", type=str, default='bf16', help="bf16 or fp16")
    parser.add_argument("--num_frames", type=int, default=32, help="num_frames: 32 or 128")
    parser.add_argument("--image_size", type=str, default="(720, 1280)", help="image_size: (720, 1280) or (512, 512)")
    parser.add_argument("--fps", type=int, default=8, help="fps: 8")
    parser.add_argument("--enable_sequence_parallelism", type=bool, default=False, help="enable_sequence_parallelism")
    parser.add_argument("--set_patch_parallel", type=bool, default=False, help="set_patch_parallel")
    parser.add_argument("--test_acc", action="store_true", help="Run or not.")
    parser.add_argument(
        "--path",
        type=str,
        default='/open-sora',
        help="The path of all model weights, suach as vae, transformer, text_encoder, tokenizer, scheduler",
    )
    parser.add_argument(
        "--prompts",
        type=list,
        default=[
            'A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. \
             She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. \
             She wears sunglasses and red lipstick. She walks confidently and casually. \
             The street is damp and reflective, creating a mirror effect of the colorful lights. \
             Many pedestrians walk about.'],
        help="prompts",
    )
    return parser.parse_args()


def get_dtype(args):
    dtype = torch.bfloat16
    if args.type == 'bf16':
        dtype = torch.bfloat16
    elif args.type == 'fp16':
        dtype = torch.float16
    else:
        logger.error("Not supported.")
    return dtype


def get_prompts(args, is_test_acc):
    if not is_test_acc:
        prompts = args.prompts
    else:
        lines_list = []
        with open('./tests/t2v_sora.txt', 'r') as file:
            for line in file:
                line = line.strip()
                lines_list.append(line)
        prompts = lines_list
    return prompts


def get_video(index, pipe, prompts, is_test_acc):
    if is_test_acc:
        video = pipe(prompts=[prompts[index]], output_type="thwc")

    else:
        video = pipe(prompts=prompts)
    return video


def infer(args):
    test_acc = args.test_acc
    use_time = 0
    torch.npu.set_device(args.device_id)
    dtype = get_dtype(args)

    # === Initialize Distributed ===
    if args.enable_sequence_parallelism or args.set_patch_parallel:
        init_parallel_env(args.enable_sequence_parallelism or args.set_patch_parallel)

    args.image_size = eval(args.image_size)

    prompts = get_prompts(args, test_acc)

    if not test_acc:
        loops = 5
    else:
        loops = len(prompts)

    pipe = OpenSoraPipeline12.from_pretrained(model_path=args.path,
                                              num_frames=args.num_frames, image_size=args.image_size, fps=args.fps,
                                              enable_sequence_parallelism=args.enable_sequence_parallelism,
                                              dtype=dtype)
    pipe = compile_pipe(pipe)

    for i in range(loops):

        start_time = time.time()
        video = get_video(i, pipe, prompts, test_acc)
        torch.npu.empty_cache()

        if test_acc:
            save_file_name = "sample_{:0>2d}.mp4".format(i)
            save_path = os.path.join(os.getcwd(), save_file_name)

            write_video(save_path, video, fps=8, video_codec="h264")
            torch.npu.empty_cache()
        else:
            if i >= 2:
                use_time += time.time() - start_time
                logger.info("current_time is %.3f )", time.time() - start_time)
        torch.npu.empty_cache()

    if not test_acc:
        logger.info("use_time is %.3f)", use_time / 3)


if __name__ == "__main__":
    inference_args = parse_arguments()
    infer(inference_args)

