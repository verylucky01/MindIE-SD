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
from transformers import T5EncoderModel, AutoTokenizer
from opensora import VideoAutoencoder, RFlowScheduler, STDiT3
from mindiesd import OpenSoraPipeline12, OpenSoraPipeline
from mindiesd.utils import ParametersInvalid


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default='./open-sora',
        help="The path of all model weights, suach as vae, transformer, text_encoder, tokenizer, scheduler",
    )
    return parser.parse_args()


class TestOpenSoraPipelineFromPretrained(unittest.TestCase):

    def setUp(self):
        args = parse_arguments()
        self.model_path = args.path

    def test_from_pretrained(self):

        pipeline = OpenSoraPipeline12.from_pretrained(
            model_path=self.model_path,
            num_frames=32, image_size=(720, 1280), fps=8,
            enable_sequence_parallelism=False,
            dtype=torch.bfloat16)

        self.assertIsInstance(pipeline, OpenSoraPipeline)

        self.assertIsInstance(pipeline.opensora_pipeline.vae, VideoAutoencoder)
        self.assertIsInstance(pipeline.opensora_pipeline.text_encoder, T5EncoderModel)
        self.assertIsInstance(pipeline.opensora_pipeline.transformer, STDiT3)
        self.assertIsInstance(pipeline.opensora_pipeline.scheduler, RFlowScheduler)

    def test_invalid_model_path(self):
        with self.assertRaises(ParametersInvalid):
            OpenSoraPipeline12.from_pretrained("/invalid/path")

    def test_missing_required_module(self):
        with self.assertRaises(ValueError):
            OpenSoraPipeline12.from_pretrained(self.model_path, vae=None)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'])