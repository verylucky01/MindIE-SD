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
#include "matmulv2.h"

using namespace at;

at::Tensor matmulv2_mindie_sd_impl_npu(
    const at::Tensor &input_x1, const at::Tensor &input_x2,
    const c10::optional<at::Tensor> &bias_opt,
    const c10::optional<at::Tensor> &offset_w_opt,
    const c10::optional<bool> &transpose_x1_opt,
    const c10::optional<bool> &transpose_x2_opt,
    const c10::optional<int64_t> &offset_x_opt)
{
    size_t x1_dim = input_x1.sizes().size();
    if (x1_dim != 2) { // 2 is the first input dimension
        throw std::invalid_argument("The first input dimension of Matmul must be 2 but got " + str(x1_dim));
    }
    size_t x2_dim = input_x2.sizes().size();
    if (x2_dim != 2) { // 2 is the second input dimension
        throw std::invalid_argument("The second input dimension of Matmul must be 2 but got " + str(x2_dim));
    }
    bool transpose_x1 = transpose_x1_opt.value_or(false);
    bool transpose_x2 = transpose_x2_opt.value_or(false);
    int64_t offset_x = offset_x_opt.value_or(0);
    const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
    const at::Tensor& offset_w = c10::value_or_else(offset_w_opt, [] {return at::Tensor();});

    at::Tensor output_y = at::Tensor();
    if (transpose_x2) {
        output_y =
            at_npu::native::empty_with_format({input_x1.sizes()[0], input_x2.sizes()[0]},
            input_x1.options(), at_npu::native::get_npu_format(input_x1));
    } else {
        output_y =
            at_npu::native::empty_with_format({input_x1.sizes()[0], input_x2.sizes()[1]},
            input_x1.options(), at_npu::native::get_npu_format(input_x1));
    }

    at_npu::native::OpCommand cmd;

    cmd.Name("MatMulV2")
            .Input(input_x1, "input_x1")
            .Input(input_x2, "input_x2");

    if (bias.defined()) {
        cmd.Input(bias, "bias");
    }

    if (offset_w.defined()) {
        cmd.Input(offset_w, "offset_w");
    }

    cmd.Output(output_y, "output_y");

    if (transpose_x1) {
        cmd.Attr("transpose_x1", transpose_x1);
    }

    if (transpose_x2) {
        cmd.Attr("transpose_x2", transpose_x2);
    }

    if (offset_x != 0) {
        cmd.Attr("offset_x", offset_x);
    }

    cmd.Run();

    return output_y;
}