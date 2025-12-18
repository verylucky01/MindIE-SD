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
import unittest
import torch
from packaging.version import Version

from mindiesd.layers._custom_ops import (
    rope,
    laser_attention,
    laser_attention_preprocess,
    batch_matmul_v2,
    batch_matmul_v3,
    batch_matmul_v3_duo,
    matmul_v2
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from mindiesd.compilation import MindieSDBackend


class TestCustomOps(unittest.TestCase):
    
    def test_rope_fake_shape(self):
       
        class RopeModel(torch.nn.Module):
            def forward(self, x, cos, sin, mode):
                return rope(x, cos, sin, mode)
        
        x = torch.randn(2, 64, 8, 16, dtype=torch.bfloat16, device="npu")
        cos = torch.randn(1, 64, 1, 16, dtype=torch.bfloat16, device="npu")
        sin = torch.randn(1, 64, 1, 16, dtype=torch.bfloat16, device="npu")
        mode = 0
        
        model = RopeModel()
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        
        output_original = model(x, cos, sin, mode)
        output_compiled = compiled_model(x, cos, sin, mode)
        
        self.assertEqual(output_original.shape, output_compiled.shape)
    
    def test_laser_attention_fake_shape(self):
        
        class LaserAttentionModel(torch.nn.Module):
            def forward(self, 
                query, key, value,
                atten_mask, alibi_mask, drop_mask,
                scale_value, head_num, input_layout,
                keep_prob, pre_tokens, next_tokens,
                is_high_precision):

                return laser_attention(
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
                    is_high_precision=is_high_precision,
                )[0]
        
        batch_size = 2
        seq_len = 256
        head_num = 8
        head_dim = 128
        
        query = torch.randn(batch_size, head_num, seq_len, head_dim, dtype=torch.float16, device="npu")
        key = torch.randn(batch_size, head_num, seq_len, head_dim, dtype=torch.float16, device="npu")
        value = torch.randn(batch_size, head_num, seq_len, head_dim, dtype=torch.float16, device="npu")

        layout = "BNSD"
        pre_tokens = 0

        scale_value = 1.0
        keep_prob = 1.0
        input_layout = layout
        is_high_precision = True
        next_tokens = 1
        
        atten_mask = None
        alibi_mask = None
        drop_mask = None
        model = LaserAttentionModel()
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        
        output_original = model(
            query, key, value,
            atten_mask, alibi_mask, drop_mask,
            scale_value, head_num, input_layout,
            keep_prob, pre_tokens, next_tokens,
            is_high_precision)
        output_compiled = compiled_model(
            query, key, value,
            atten_mask, alibi_mask, drop_mask,
            scale_value, head_num, input_layout,
            keep_prob, pre_tokens, next_tokens,
            is_high_precision)
        
        self.assertEqual(output_original.shape, output_compiled.shape)
    
    def test_laser_attention_preprocess_fake_shape(self):
        
        
        class LaserAttentionPreprocessModel(torch.nn.Module):
            def forward(self, query, key, value, align_len):
                return laser_attention_preprocess(query, key, value, align_len)
        
        batch_size = 2
        seq_len = 64
        head_num = 8
        head_dim = 16
        align_len = 32
        
        query = torch.randn(batch_size, seq_len, head_num, head_dim, dtype=torch.float16, device="npu")
        key = torch.randn(batch_size, seq_len, head_num, head_dim, dtype=torch.float16, device="npu")
        value = torch.randn(batch_size, seq_len, head_num, head_dim, dtype=torch.float16, device="npu")
        
        model = LaserAttentionPreprocessModel()
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        
        output_original = model(query, key, value, align_len)
        output_compiled = compiled_model(query, key, value, align_len)
        
        self.assertEqual(len(output_original), len(output_compiled))
        for orig, comp in zip(output_original, output_compiled):
            self.assertEqual(orig.shape, comp.shape)
    
    def test_batch_matmul_v2_fake_shape(self):
        
        
        class BatchMatmulV2Model(torch.nn.Module):
            def forward(self, input_x1, input_x2, bias):
                return batch_matmul_v2(input_x1, input_x2, bias=bias)
        
        batch_size = 2
        m = 4
        k = 8
        n = 6
        
        input_x1 = torch.randn(batch_size, m, k, dtype=torch.float32, device="npu")
        input_x2 = torch.randn(batch_size, k, n, dtype=torch.float32, device="npu")
        bias = torch.randn(n, dtype=torch.float32, device="npu")
        
        model = BatchMatmulV2Model()
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        
        output_original = model(input_x1, input_x2, bias)
        output_compiled = compiled_model(input_x1, input_x2, bias)
        
        self.assertEqual(output_original.shape, output_compiled.shape)
    
    def test_batch_matmul_v3_fake_shape(self):
        
        
        class BatchMatmulV3Model(torch.nn.Module):
            def forward(self, x1, x2, bias):
                return batch_matmul_v3(x1, x2, bias=bias, enable_hf32=False)
        
        batch_size = 2
        m = 4
        k = 8
        n = 6
        
        x1 = torch.randn(batch_size, m, k, dtype=torch.float32, device="npu")
        x2 = torch.randn(batch_size, k, n, dtype=torch.float32, device="npu")
        bias = torch.randn(n, dtype=torch.float32, device="npu")
        
        model = BatchMatmulV3Model()
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        
        output_original = model(x1, x2, bias)
        output_compiled = compiled_model(x1, x2, bias)
        
        self.assertEqual(output_original.shape, output_compiled.shape)
    
    def test_batch_matmul_v3_duo_fake_shape(self):
       
        class BatchMatmulV3DuoModel(torch.nn.Module):
            def forward(self, x1, x2):
                return batch_matmul_v3_duo(x1, x2)
        
        batch_size = 2
        m = 4
        k = 8
        n = 6
        
        x1 = torch.randn(batch_size, m, k, dtype=torch.float32, device="npu")
        x2 = torch.randn(batch_size, k, n, dtype=torch.float32, device="npu")
        
        model = BatchMatmulV3DuoModel()
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        
        output_original = model(x1, x2)
        output_compiled = compiled_model(x1, x2)
        
        self.assertEqual(output_original.shape, output_compiled.shape)
    
    def test_matmul_v2_fake_shape(self):
        
        class MatmulV2Model(torch.nn.Module):
            def forward(self, input_x1, input_x2, bias):
                return matmul_v2(input_x1, input_x2, bias=bias)
        
        m = 4
        k = 8
        n = 6
        
        input_x1 = torch.randn(m, k, dtype=torch.float32, device="npu")
        input_x2 = torch.randn(k, n, dtype=torch.float32, device="npu")
        bias = torch.randn(n, dtype=torch.float32, device="npu")
        
        model = MatmulV2Model()
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        
        output_original = model(input_x1, input_x2, bias)
        output_compiled = compiled_model(input_x1, input_x2, bias)
        
        self.assertEqual(output_original.shape, output_compiled.shape)

if __name__ == '__main__':
    unittest.main()