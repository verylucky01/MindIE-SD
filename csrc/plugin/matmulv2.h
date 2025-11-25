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

#ifndef MATMULV2_MINDIE_SD_IMPL_H
#define MATMULV2_MINDIE_SD_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

at::Tensor matmulv2_mindie_sd_impl_npu(
    const at::Tensor &input_x1, const at::Tensor &input_x2,
    const c10::optional<at::Tensor> &bias_opt,
    const c10::optional<at::Tensor> &offset_w_opt,
    const c10::optional<bool> &transpose_x1_opt,
    const c10::optional<bool> &transpose_x2_opt,
    const c10::optional<int64_t> &offset_x_opt);
#endif // MATMULV2_MINDIE_SD_IMPL_H