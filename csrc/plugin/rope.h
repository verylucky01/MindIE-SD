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

#ifndef ROPE_MINDIE_SD_IMPL_H
#define ROPE_MINDIE_SD_IMPL_H

#include <ATen/Tensor.h>

at::Tensor rope_mindie_sd_impl_npu(const at::Tensor &x, const at::Tensor &cos, const at::Tensor &sin, int64_t mode);

#endif // ROPE_MINDIE_SD_IMPL_H

