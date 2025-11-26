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

#include "register/register.h"

namespace domi {
    // Onnx ParseParams
    Status ParseParamFlashAttentionTik(const ge::Operator& opSrc, ge::Operator& opDest)
    {
        (void) opSrc;
        (void) opDest;
        return SUCCESS;
    }

    static std::vector<ge::AscendString> supportedOnnxVersion ({
        "ai.onnx::8::FlashAttentionTik",
        "ai.onnx::9::FlashAttentionTik",
        "ai.onnx::10::FlashAttentionTik",
        "ai.onnx::11::FlashAttentionTik",
        "ai.onnx::12::FlashAttentionTik",
        "ai.onnx::13::FlashAttentionTik",
        "ai.onnx::14::FlashAttentionTik",
        "ai.onnx::15::FlashAttentionTik",
        "ai.onnx::16::FlashAttentionTik",
        "ai.onnx::17::FlashAttentionTik",
        "ai.onnx::18::FlashAttentionTik",
    });

    // register FlashAttentionTik op info to GE
    REGISTER_CUSTOM_OP("FlashAttentionTik")
        .FrameworkType(ONNX)
        .OriginOpType(supportedOnnxVersion)
        .ParseParamsByOperatorFn(ParseParamFlashAttentionTik)
        .ImplyType(ImplyType::TVM);
}  // namespace domi
