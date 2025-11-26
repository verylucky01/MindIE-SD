/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */


#include <torch/library.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "rope.h"

using namespace at;

at::Tensor rope_mindie_sd_impl_npu(const at::Tensor &x, const at::Tensor &cos, const at::Tensor &sin, int64_t mode = 1)
{
    at::Tensor result = at_npu::native::empty_with_format(x.sizes(), x.options(), at_npu::native::get_npu_format(x));

    at_npu::native::OpCommand cmd;

    cmd.Name("RotaryPositionEmbedding")
            .Input(x)
            .Input(cos)
            .Input(sin)
            .Output(result)
            .Attr("mode", mode)
            .Run();

    return result;
}