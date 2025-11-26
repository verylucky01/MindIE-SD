/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
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
    Status ParseParamAscendLaserAttention(const ge::Operator& opSrc, ge::Operator& opDest)
    {
        (void) opSrc;
        (void) opDest;
        return SUCCESS;
    }

    static std::vector<ge::AscendString> supportedOnnxVersion ({
        "ai.onnx::8::AscendLaserAttention",
        "ai.onnx::9::AscendLaserAttention",
        "ai.onnx::10::AscendLaserAttention",
        "ai.onnx::11::AscendLaserAttention",
        "ai.onnx::12::AscendLaserAttention",
        "ai.onnx::13::AscendLaserAttention",
        "ai.onnx::14::AscendLaserAttention",
        "ai.onnx::15::AscendLaserAttention",
        "ai.onnx::16::AscendLaserAttention",
        "ai.onnx::17::AscendLaserAttention",
        "ai.onnx::18::AscendLaserAttention",
    });

    // register AscendLaserAttention op info to GE
    REGISTER_CUSTOM_OP("AscendLaserAttention")
        .FrameworkType(ONNX)
        .OriginOpType(supportedOnnxVersion)
        .ParseParamsByOperatorFn(ParseParamAscendLaserAttention)
        .ImplyType(ImplyType::TVM);
}  // namespace domi
