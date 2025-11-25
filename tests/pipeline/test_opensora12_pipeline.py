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
sys.path.append('../')

import unittest
from unittest.mock import MagicMock, patch
from transformers import AutoTokenizer, T5EncoderModel
from mindiesd.models import STDiT3, VideoAutoencoder
from mindiesd.schedulers import RFlowScheduler
from mindiesd.pipeline.open_sora_pipeline import OpenSoraPipeline12, MAX_PROMPT_LENGTH
from mindiesd.utils import ModelInitError


class TestOpenSoraPipeline12(unittest.TestCase):
    def test_init(self):
        text_encoder = MagicMock(spec=T5EncoderModel)
        tokenizer = MagicMock(spec=AutoTokenizer)
        transformer = MagicMock(spec=STDiT3)
        vae = MagicMock(spec=VideoAutoencoder)
        scheduler = MagicMock(spec=RFlowScheduler)

        with patch.dict('sys.modules', {'opensora': None}):
            with self.assertRaises(ModelInitError) as context:
                pipe = OpenSoraPipeline12(text_encoder, tokenizer, transformer, vae, scheduler)
            self.assertIn("Failed to find the opensora library", str(context.exception))


if __name__ == '__main__':
    unittest.main()