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
#include "batchmatmulv3duo.h"

using namespace at;

at::Tensor batchmatmulv3duo_mindie_sd_impl_npu(
    const at::Tensor &x1, const at::Tensor &x2,
    const c10::optional<at::Tensor> &bias_opt,
    const c10::optional<at::Tensor> &offset_w_opt)
{
    size_t x1_dim = x1.sizes().size();
    if (x1_dim != 3) {  // 3: batchmatmulv3duo input dim
        throw std::invalid_argument("The first input dimension of BatchMatmulv3duo must be 3 but got " + str(x1_dim));
    }

    size_t x2_dim = x2.sizes().size();
    if (x2_dim != 3) {  // batchmatmulv3duo input dim
        throw std::invalid_argument("The second input dimension of BatchMatmulv3duo must be 3 but got " + str(x2_dim));
    }

    const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
    const at::Tensor& offset_w = c10::value_or_else(offset_w_opt, [] {return at::Tensor();});

    at::Tensor y =
        at_npu::native::empty_with_format({x1.sizes()[0], x1.sizes()[1], x2.sizes()[2]},
        x1.options(), at_npu::native::get_npu_format(x1));

    at_npu::native::OpCommand cmd;

    cmd.Name("BatchMatMulV3")
            .Input(x1, "x1")
            .Input(x2, "x2");

    if (bias.defined()) {
        cmd.Input(bias, "bias");
    }
    if (offset_w.defined()) {
        cmd.Input(offset_w, "offset_w");
    }

    cmd.Output(y, "y");

    cmd.Run();

    return y;
}