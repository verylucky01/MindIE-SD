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


import logging
import unittest
import sys
import os
import json
from unittest.mock import patch
from diffusers.models import AutoencoderKL
from safetensors.torch import save_file
import torch.nn as nn

sys.path.append('../')
from mindiesd.models import VideoAutoencoder
from mindiesd.models.vae import VideoAutoencoderConfig
from mindiesd.utils.file_utils import MODELDATA_FILE_PERMISSION
from mindiesd.utils import ModelInitError

DOWN_ENCODER_BLOCK2D = "DownEncoderBlock2D"
UP_DECODER_BLOCK2D = "UpDecoderBlock2D"


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TestSpatialVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_vae = AutoencoderKL(
            act_fn="silu",
            block_out_channels=[
                128,
                256,
                512,
                512
            ],
            down_block_types=[
                DOWN_ENCODER_BLOCK2D,
                DOWN_ENCODER_BLOCK2D,
                DOWN_ENCODER_BLOCK2D,
                DOWN_ENCODER_BLOCK2D
            ],
            in_channels=3,
            latent_channels=4,
            layers_per_block=2,
            norm_num_groups=32,
            out_channels=3,
            sample_size=512,
            scaling_factor=0.13025,
            up_block_types=[
                UP_DECODER_BLOCK2D,
                UP_DECODER_BLOCK2D,
                UP_DECODER_BLOCK2D,
                UP_DECODER_BLOCK2D
            ],
            force_upcast=False
        )


class TestVideoAutoencoder(unittest.TestCase):

    def setUp(self):
        spatial_vae = TestSpatialVAE().spatial_vae
        save_file(spatial_vae.state_dict(), "test_models/vae/vae_2d/vae/diffusion_pytorch_model.safetensors")
        os.chmod("test_models/vae/vae_2d/vae/diffusion_pytorch_model.safetensors", MODELDATA_FILE_PERMISSION)
    
    def test_init(self):
        config = VideoAutoencoderConfig(from_pretrained="test_models/vae", set_patch_parallel=False)
        with patch.dict('sys.modules', {'opensora': None}):
            with self.assertRaises(ModelInitError) as context:
                VideoAutoencoder(config)
            self.assertIn("Failed to find the opensora library", str(context.exception))
    
    def test_invalid_config(self):
        try:
            config = VideoAutoencoderConfig(from_pretrained="test_models/vae")
            invalid_path = "invalid_path"
            config.vae_2d['from_pretrained'] = invalid_path
            vae = VideoAutoencoder(config)
        except Exception as e:
            logger.error(e)
            config = None
        self.assertIsNone(config)

if __name__ == '__main__':
    unittest.main()