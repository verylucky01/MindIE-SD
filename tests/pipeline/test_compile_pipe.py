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
from unittest.mock import patch, MagicMock
from mindiesd.utils import is_npu_available
from mindiesd import compile_pipe

NPU = 'npu'


class TestCompilePipe(unittest.TestCase):

    @patch('mindiesd.utils.is_npu_available', return_value=True)
    def test_compile_pipe_npu_not_available(self, mock_is_npu_available):
        # Mocking the pipe object
        pipe = MagicMock()

        # Call the function and expect a RuntimeError
        compile_pipe(pipe)

        # Assertions
        pipe.text_encoder.to.assert_not_called()
        pipe.transformer.to.assert_not_called()
        pipe.vae.to.assert_not_called()

if __name__ == '__main__':
    unittest.main()