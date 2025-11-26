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
import os
import sys
sys.path.append('../')

import torch
import torch.nn as nn
from safetensors.torch import save_file
from device import DEVICE_ID
from mindiesd.models.model_utils import DiffusionModel
from mindiesd.config_utils import ConfigMixin
from mindiesd.utils.file_utils import MODELDATA_FILE_PERMISSION
from mindiesd.utils.logs.logging import logger


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope
        
        self.is_causal = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_x_shape0, n_x_shape1, c_x_shape2 = x.shape 
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (n_x_shape1 > b_x_shape0)
        qkv = self.qkv(x)
        qkv_shape = (b_x_shape0, n_x_shape1, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.is_causal,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            if self.is_causal:
                causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
                causal_mask = torch.where(causal_mask.bool(), 0, float('-inf'))
                attn += causal_mask
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (b_x_shape0, n_x_shape1, c_x_shape2)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention(1280, 16, False, False)
    
    def forward(self, x):
        return self.attention(x)


class ModelConfig(ConfigMixin):
    config_name = "config.json"

    def __init__(self, dimension, num_heads, qkv_bias, qk_norm):
        self.dimension = dimension
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm


class TestModelPth(DiffusionModel):
    config_class = ModelConfig
    weights_name = "./model.pth"

    def __init__(self, config):
        super().__init__(config)
        self.attention = Attention(
            config.dimension, config.num_heads, config.qkv_bias, config.qk_norm)
    
    def forward(self, x):
        return self.attention(x)

    def load_weights(self, state_dict, shard=False):
        with torch.no_grad():
            if not shard:
                self.load_state_dict(state_dict)
                return {}
            else:
                self.load_state_dict(state_dict, strict=False)
                return state_dict.keys()


class TestModelSafetensors(DiffusionModel):
    config_class = ModelConfig
    weights_name = "./model.safetensors"

    def __init__(self, config):
        super().__init__(config)
        self.attention = Attention(
            config.dimension, config.num_heads, config.qkv_bias, config.qk_norm)
    
    def forward(self, x):
        return self.attention(x)

    def load_weights(self, state_dict, shard=False):
        with torch.no_grad():
            if not shard:
                self.load_state_dict(state_dict)
                return {}
            else:
                self.load_state_dict(state_dict, strict=False)
                return state_dict.keys()


class TestModelInvalidPth(DiffusionModel):
    config_class = ModelConfig
    weights_name = "./invalid_model.pth"

    def __init__(self, config):
        super().__init__(config)
        self.attention = Attention(
            config.dimension, config.num_heads, config.qkv_bias, config.qk_norm)
    
    def forward(self, x):
        return self.attention(x)


class TestModelInvalidSafetensors(DiffusionModel):
    config_class = ModelConfig
    weights_name = "./invalid_model.safetensors"

    def __init__(self, config):
        super().__init__(config)
        self.attention = Attention(
            config.dimension, config.num_heads, config.qkv_bias, config.qk_norm)
    
    def forward(self, x):
        return self.attention(x)


class ModelConfigInvalid(ConfigMixin):
    config_name = "invalid_config.json"

    def __init__(self, dimension, num_heads, qkv_bias, qk_norm):
        self.dimension = dimension
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm


class TestModelConfigInvalid(DiffusionModel):
    config_class = ModelConfigInvalid
    weights_name = "./model.pth"

    def __init__(self, config):
        super().__init__(config)
        self.attention = Attention(
            config.dimension, config.num_heads, config.qkv_bias, config.qk_norm)
    
    def forward(self, x):
        return self.attention(x)


class TestDiffusionModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234) # 1234: rand seed
        self.model = TestModel()
        torch.save(self.model.state_dict(), "./test_models/model.pth")
        os.chmod("./test_models/model.pth", MODELDATA_FILE_PERMISSION)
        torch.save(self.model.state_dict(), "./test_models/invalid_model.safetensors")
        os.chmod("./test_models/invalid_model.safetensors", MODELDATA_FILE_PERMISSION)
        save_file(self.model.state_dict(), "./test_models/model.safetensors")
        os.chmod("./test_models/model.safetensors", MODELDATA_FILE_PERMISSION)
        save_file(self.model.state_dict(), "./test_models/invalid_model.pth")
        os.chmod("./test_models/invalid_model.pth", MODELDATA_FILE_PERMISSION)

    def test_from_pretrained_pth(self):
        test_model = TestModelPth.from_pretrained("./test_models", dtype=torch.float16)
        inputs = torch.randn([1, 64, 1280], dtype=torch.float16)
        import torch_npu
        torch_npu.npu.set_device(DEVICE_ID)
        device = "npu"
        test_model.to(device)
        self.model.to(device)
        inputs = inputs.to(device)
        test_result = test_model.forward(inputs).reshape(1, -1)
        result = self.model.forward(inputs).reshape(1, -1)
        self.assertGreater(torch.cosine_similarity(test_result, result)[0], 0.999)
    
    def test_from_pretrained_safetensors(self):
        test_model = TestModelSafetensors.from_pretrained("./test_models", dtype=torch.float16)
        inputs = torch.randn([1, 64, 1280], dtype=torch.float16)
        import torch_npu
        torch_npu.npu.set_device(DEVICE_ID)
        device = "npu"
        test_model.to(device)
        self.model.to(device)
        inputs = inputs.to(device)
        test_result = test_model.forward(inputs).reshape(1, -1)
        result = self.model.forward(inputs).reshape(1, -1)
        self.assertGreater(torch.cosine_similarity(test_result, result)[0], 0.999)

    def test_from_pretrained_invalid_config(self):
        try:
            test_model = TestModelConfigInvalid.from_pretrained("./test_models", dtype=torch.float16)
        except Exception as e:
            logger.error(e)
            test_model = None
        self.assertIsNone(test_model)

    def test_from_pretrained_invalid_pth(self):
        try:
            test_model = TestModelInvalidPth.from_pretrained("./test_models", dtype=torch.float16)
        except Exception as e:
            logger.error(e)
            test_model = None
        self.assertIsNone(test_model)

    def test_from_pretrained_invalid_safetensors(self):
        try:
            test_model = TestModelInvalidSafetensors.from_pretrained("./test_models", dtype=torch.float16)
        except Exception as e:
            logger.error(e)
            test_model = None
        self.assertIsNone(test_model)


if __name__ == '__main__':
    unittest.main()