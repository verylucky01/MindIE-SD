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

#include <vector>
#include <algorithm>

#include "flash_attention_tik.h"
namespace {
    constexpr int64_t MAX_HEAD_NUM = 4096;
    constexpr int64_t MAX_SEQLEN = 131072;
    constexpr int64_t MAX_BATCH_HEAD = 64;
    constexpr int64_t MIN_BATCH = 1;
    constexpr int64_t MAX_BATCH = 64;
    const std::vector<int64_t> HEAD_DIM_LIST = { 40, 48, 64, 72, 80, 128, 160, 512 };
}
namespace ge {
static bool CheckValueRange(int64_t value, int64_t minVal, int64_t maxVal)
{
    return ((value >= minVal) && (value <= maxVal));
}

static bool CheckHeadDim(int64_t qHeadDim, int64_t kHeadDim, int64_t vHeadDim)
{
    if (std::find(HEAD_DIM_LIST.begin(), HEAD_DIM_LIST.end(), qHeadDim) == HEAD_DIM_LIST.end()) {
        return false;
    }
    if ((kHeadDim != qHeadDim) || (vHeadDim != qHeadDim)) {
        return false;
    }
    return true;
}

static bool CheckSeqlen(int64_t qSeqlen, int64_t kSeqlen, int64_t vSeqlen)
{
    if (!CheckValueRange(qSeqlen, 1, MAX_SEQLEN) || !CheckValueRange(kSeqlen, 1, MAX_SEQLEN) ||
        (vSeqlen != kSeqlen)) {
        return false;
    }
    return true;
}

static bool CheckBatch(int64_t qBatch, int64_t kBatch, int64_t vBatch)
{
    if (!CheckValueRange(qBatch, MIN_BATCH, MAX_BATCH) || (kBatch != qBatch) || (vBatch != qBatch)) {
        return false;
    }
    return true;
}

static bool CheckHeadNum(int64_t qHead, int64_t kHead, int64_t vHead)
{
    if (!CheckValueRange(qHead, 1, MAX_HEAD_NUM) || (kHead != qHead) || (vHead != qHead)) {
        return false;
    }
    return true;
}

static bool CheckBatchHead(int64_t qBatchHead, int64_t kBatchHead, int64_t vBatchHead)
{
    if (!CheckValueRange(qBatchHead, 1, MAX_BATCH_HEAD) || (kBatchHead != qBatchHead) ||
        (vBatchHead != qBatchHead)) {
        return false;
    }
    return true;
}

static ge::graphStatus CheckInputValShape3(std::vector<int64_t> &q, std::vector<int64_t> &k, std::vector<int64_t> &v)
{
    // case 0: ori shape is 3
    if ((q.size() == 3) && (k.size() == 3) && (v.size() == 3)) {
        int64_t qBatchHead = q[0];
        int64_t kBatchHead = k[0];
        int64_t vBatchHead = v[0];
        if (!CheckBatchHead(qBatchHead, kBatchHead, vBatchHead)) {
            return GRAPH_FAILED;
        }

        // 1 for seqlen pos in q shape
        int64_t qSeqlen = q[1];
        // 1 for seqlen pos in k shape
        int64_t kSeqlen = k[1];
        // 1 for seqlen pos in v shape
        int64_t vSeqlen = v[1];
        if (!CheckSeqlen(qSeqlen, kSeqlen, vSeqlen)) {
            return GRAPH_FAILED;
        }

        // 2 for head dim pos in q shape
        int64_t qHeadDim = q[2];
        // 2 for head dim pos in k shape
        int64_t kHeadDim = k[2];
        // 2 for head dim pos in v shape
        int64_t vHeadDim = v[2];
        if (!CheckBatchHead(qHeadDim, kHeadDim, vHeadDim)) {
            return GRAPH_FAILED;
        }
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputValShape4(std::vector<int64_t> &q, std::vector<int64_t> &k, std::vector<int64_t> &v)
{
    // case 1: ori shape is 4
    if ((q.size() == 4) && (k.size() == 4) && (v.size() == 4)) {
        int64_t qBatch = q[0];
        int64_t kBatch = k[0];
        int64_t vBatch = v[0];
        if (!CheckBatch(qBatch, kBatch, vBatch)) {
            return GRAPH_FAILED;
        }

        // 1 for head num pos in q shape
        int64_t qHeadNum = q[1];
        // 1 for head num pos in k shape
        int64_t kHeadNum = k[1];
        // 1 for head num pos in v shape
        int64_t vHeadNum = v[1];
        if (!CheckHeadNum(qHeadNum, kHeadNum, vHeadNum)) {
            return GRAPH_FAILED;
        }

        // 2 for seqlen pos in q shape
        int64_t qSeqlen = q[2];
        // 2 for seqlen pos in k shape
        int64_t kSeqlen = k[2];
        // 2 for seqlen pos in v shape
        int64_t vSeqlen = v[2];
        if (!CheckSeqlen(qSeqlen, kSeqlen, vSeqlen)) {
            return GRAPH_FAILED;
        }

        // 3 for head dim pos in q shape
        int64_t qHeadDim = q[3];
        // 3 for head dim pos in k shape
        int64_t kHeadDim = k[3];
        // 3 for head dim pos in v shape
        int64_t vHeadDim = v[3];
        if (!CheckHeadDim(qHeadDim, kHeadDim, vHeadDim)) {
            return GRAPH_FAILED;
        }
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FlashAttentionTikInferShape)
{
    // check input number is 3 and output number is 1
    if ((op.GetInputsSize() != 3) || (op.GetOutputsSize() != 1)) {
        return GRAPH_FAILED;
    }

    std::vector<int64_t> qDims = op.GetInputDescByName("q").GetShape().GetDims();
    std::vector<int64_t> kDims = op.GetInputDescByName("k").GetShape().GetDims();
    std::vector<int64_t> vDims = op.GetInputDescByName("v").GetShape().GetDims();
    // check ori shape should be 3 or 4
    if ((qDims.size() != 3) && (qDims.size() != 4)) {
        return GRAPH_FAILED;
    }
    if ((kDims.size() != qDims.size()) || (vDims.size() != qDims.size())) {
        return GRAPH_FAILED;
    }

    // check value for shape dims 3
    if ((qDims.size() == 3) && (CheckInputValShape3(qDims, kDims, vDims) == GRAPH_FAILED)) {
        return GRAPH_FAILED;
    }
    // check value for shape dims 4
    if ((qDims.size() == 4) && (CheckInputValShape4(qDims, kDims, vDims) == GRAPH_FAILED)) {
        return GRAPH_FAILED;
    }

    TensorDesc tensordescOutput = op.GetOutputDescByName("y");
    tensordescOutput.SetShape(op.GetInputDescByName("q").GetShape());
    tensordescOutput.SetDataType(op.GetInputDescByName("q").GetDataType());
    tensordescOutput.SetFormat(op.GetInputDescByName("q").GetFormat());
    (void)op.UpdateOutputDesc("y", tensordescOutput);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(FlashAttentionTik, FlashAttentionTikVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FlashAttentionTik, FlashAttentionTikInferShape);
VERIFY_FUNC_REG(FlashAttentionTik, FlashAttentionTikVerify);

}  // namespace ge
