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

#include "rope.h"
#include "la.h"
#include "matmulv2.h"
#include "batchmatmulv3.h"
#include "batchmatmulv3duo.h"
#include "batchmatmulv2.h"
#include "adalayernorm.h"
#include "la_preprocess.h"


TORCH_LIBRARY(mindie, m)
{
    m.def("rope_mindie_sd(Tensor query, Tensor key, Tensor value, int mode) -> Tensor");
    m.def(
        "la_mindie_sd(Tensor query, Tensor key, Tensor value, \
        Tensor? atten_mask=None, Tensor? alibi_mask=None, Tensor? \
        drop_mask=None, float scale_value=1.0, int head_num=2, str input_layout='BNSD', \
        float keep_prob=1.0, int pre_tokens=2147483647, int next_tokens=1, \
        bool is_highPrecision=True)  -> (Tensor, Tensor)");
    m.def("matmulv2_mindie_sd(Tensor input_x1, Tensor input_x2, Tensor? bias=None, Tensor? offset_w=None, \
           bool? transpose_x1=False, bool? transpose_x2=False, int? offset_x=0) \
           -> Tensor");
    m.def("batchmatmulv3_mindie_sd(Tensor x1, Tensor x2, Tensor? bias=None, Tensor? offset_w=None, \
        bool? adj_x1=False, bool? adj_x2=False, int? offset_x=0, bool? enable_hf32=False) \
        -> Tensor");
    m.def("batchmatmulv3duo_mindie_sd(Tensor x1, Tensor x2, Tensor? bias=None, Tensor? offset_w=None) \
        -> Tensor");
    m.def("batchmatmulv2_mindie_sd(Tensor input_x1, Tensor input_x2, Tensor? bias=None, Tensor? offset_w=None, \
        bool? adj_x1=False, bool? adj_x2=False, int? offset_x=0) \
        -> Tensor");
    m.def("adaln_mindie_sd(Tensor x, Tensor scale, Tensor shift, Tensor? weight=None, \
        Tensor? bias=None, float? epsilon=1e-5) \
        -> Tensor");
    m.def("la_preprocess_mindie_sd(Tensor query, Tensor key, Tensor value, int align_len=256) \
        -> (Tensor, Tensor, Tensor)");
}


TORCH_LIBRARY_IMPL(mindie, PrivateUse1, m)
{
    m.impl("rope_mindie_sd", &rope_mindie_sd_impl_npu);
    m.impl("la_mindie_sd", &la_mindie_sd_impl_npu);
    m.impl("matmulv2_mindie_sd", &matmulv2_mindie_sd_impl_npu);
    m.impl("batchmatmulv3_mindie_sd", &batchmatmulv3_mindie_sd_impl_npu);
    m.impl("batchmatmulv3duo_mindie_sd", &batchmatmulv3duo_mindie_sd_impl_npu);
    m.impl("batchmatmulv2_mindie_sd", &batchmatmulv2_mindie_sd_impl_npu);
    m.impl("adaln_mindie_sd", &adaln_mindie_sd_impl_npu);
    m.impl("la_preprocess_mindie_sd", &la_preprocess_mindie_sd_impl_npu);
}