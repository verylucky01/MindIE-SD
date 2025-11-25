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
#include "batchmatmulv2.h"

using namespace at;

at::Tensor batchmatmulv2_mindie_sd_impl_npu(
    const at::Tensor &input_x1, const at::Tensor &input_x2,
    const c10::optional<at::Tensor> &bias_opt,
    const c10::optional<at::Tensor> &offset_w_opt,
    const c10::optional<bool> &adj_x1_opt,
    const c10::optional<bool> &adj_x2_opt,
    const c10::optional<int64_t> &offset_x_opt)
{
    size_t x1_dim = input_x1.sizes().size();
    if (x1_dim != 3) {  // 3: batchmm input dim
        throw std::invalid_argument("The first input dimension of BatchMatmul must be 3 but got " + str(x1_dim));
    }
    size_t x2_dim = input_x2.sizes().size();
    if (x2_dim != 3 and x2_dim != 2) {  // batchmm input dim must be 2 or 3
        throw std::invalid_argument("The second input dimension of BatchMatmul must be 2 or 3 but got " + str(x2_dim));
    }
    bool adj_x1 = adj_x1_opt.value_or(false);
    bool adj_x2 = adj_x2_opt.value_or(false);
    int64_t offset_x = offset_x_opt.value_or(0);
    const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
    const at::Tensor& offset_w = c10::value_or_else(offset_w_opt, [] {return at::Tensor();});

    at::Tensor output_z = at::Tensor();
    int64_t index = 0;
    int64_t length = input_x2.dim();

    index = adj_x2 ? (length == 2 ? 0 : 1) : (length == 2 ? 1 : 2);    // 2、0、1 is index

    output_z = at_npu::native::empty_with_format({input_x1.sizes()[0], input_x1.sizes()[1], input_x2.sizes()[index]},
        input_x1.options(), at_npu::native::get_npu_format(input_x1));

    at_npu::native::OpCommand cmd;

    cmd.Name("BatchMatMulV2")
            .Input(input_x1, "input_x1")
            .Input(input_x2, "input_x2");

    if (bias.defined()) {
        cmd.Input(bias, "bias");
    }
    if (offset_w.defined()) {
        cmd.Input(offset_w, "offset_w");
    }

    cmd.Output(output_z, "output_z");

    if (adj_x1) {
        cmd.Attr("adj_x1", adj_x1);
    }

    if (adj_x2) {
        cmd.Attr("adj_x2", adj_x2);
    }

    if (offset_x != 0) {
        cmd.Attr("offset_x", offset_x);
    }

    cmd.Run();

    return output_z;
}
