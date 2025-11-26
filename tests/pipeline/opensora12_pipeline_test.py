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

import unittest
import argparse
import os
import torch
import torch_npu
from transformers import T5EncoderModel, AutoTokenizer
from opensora import VideoAutoencoder, RFlowScheduler, STDiT3
from mindiesd import OpenSoraPipeline12, compile_pipe


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default='./open-sora',
        help="The path of all model weights, such as vae, transformer, text_encoder, tokenizer, scheduler",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="NPU device id",
    )
    return parser.parse_args()


class TestOpenSoraPipeline12(unittest.TestCase):
    def setUp(self):
        args = parse_arguments()
        self.args = args

        self.num_frames = None
        self.image_size = None
        self.fps = None
        self.dtype = None
        self.model_path = None

        self.vae = None

        self.text_encoder = None
        self.tokenizer = None
        self.transformer = None
        self.scheduler = None

    def test_pipeline_32_512x512_bfloat16_8(self):
        self._run_pipeline(32, (512, 512), torch.bfloat16, 8)

    def test_pipeline_32_512x512_float16_8(self):
        self._run_pipeline(32, (512, 512), torch.float16, 8)

    def test_pipeline_32_720x1280_bfloat16_8(self):
        self._run_pipeline(32, (720, 1280), torch.bfloat16, 8)

    def test_pipeline_32_720x1280_float16_8(self):
        self._run_pipeline(32, (720, 1280), torch.float16, 8)

    def test_pipeline_128_512x512_bfloat16_8(self):
        self._run_pipeline(128, (512, 512), torch.bfloat16, 8)

    def test_pipeline_128_512x512_float16_8(self):
        self._run_pipeline(128, (512, 512), torch.float16, 8)

    def test_pipeline_128_720x1280_bfloat16_8(self):
        self._run_pipeline(128, (720, 1280), torch.bfloat16, 8)

    def test_pipeline_128_720x1280_float16_8(self):
        self._run_pipeline(128, (720, 1280), torch.float16, 8)

    def _init(self, model_path, num_frames, image_size, fps, dtype):
        self.num_frames = num_frames
        self.image_size = image_size
        self.fps = fps
        self.dtype = dtype
        self.model_path = model_path

        vae_path = os.path.join(model_path, "vae")
        text_encoder_path = os.path.join(model_path, "text_encoder")
        tokenizer_path = os.path.join(model_path, "tokenizer")
        transformer_path = os.path.join(model_path, "transformer")

        self.vae = VideoAutoencoder.from_pretrained(
            vae_path,
            from_pretrained=vae_path, set_patch_parallel=False).to(self.dtype)

        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_path).to(self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, dtype=self.dtype)
        self.transformer = STDiT3.from_pretrained(
            transformer_path,
            input_size=self.vae.get_latent_size((self.num_frames, *self.image_size)),
            in_channels=self.vae.out_channels,
            caption_channels=self.text_encoder.config.d_model,
            enable_flash_attn=True,
            enable_sequence_parallelism=False,
            use_cache=True,
            cache_interval=2,
            cache_start=3,
            num_cache_layer=13,
            cache_start_steps=5).to(self.dtype)
        self.scheduler = RFlowScheduler(num_timesteps=1000, num_sampling_steps=30)

    def _run_pipeline(self, num_frames, image_size, dtype, fps):
        self._init(self.args.path, num_frames, image_size, fps, dtype)

        torch.npu.set_device(self.args.device_id)
        pipeline = OpenSoraPipeline12(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
            num_frames=self.num_frames,
            image_size=self.image_size,
            fps=self.fps,
            dtype=self.dtype
        )
        pipeline = compile_pipe(pipeline.opensora_pipeline)
        prompts = ["A cat playing with a ball"]
        result = pipeline(prompts=prompts, seed=42, output_type="thwc")

        self.assertEqual(result.shape[0], self.num_frames)
        self.assertEqual(result.shape[1], self.image_size[0])
        self.assertEqual(result.shape[2], self.image_size[1])


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'])