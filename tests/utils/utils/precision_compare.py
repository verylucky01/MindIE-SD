#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import numpy as np
import torch
from mindiesd.utils.logs.logging import logger


def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def cal_relative_diff(real_data, expect_data, diff_thd, type_str='fp16'):
    if 'nan' in str(expect_data) or 'inf' in str(expect_data):
        if type_str.lower() == 'fp16':
            expect_data = 65504
        else:
            expect_data = 3.4028e38
    diff = abs(float(real_data) - float(expect_data))
    if diff < diff_thd:
        result = diff
    else:
        result = diff / (float(max(abs(real_data), abs(expect_data))) + 10e-10)
    return result


def display_output(real_data, expect_data, start, end, diff_thd):
    header_format = "{:<10s}{:<17s}{:<15s}{:<18s}{:<15s}"
    value_format_float = "{:08d}{:>15.7f}{:>15.7f}{:>15.7f}{:>15.7f}"
    value_format_str = "{:08d}{:^15s}{:^15s}{:^15s}{:^15s}"
    ellipsis_format = "{:<15s}{:^15s}{:^15s}{:^15s}{:^15s}"
    total_width = 15 * 5

    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff(expect_data[j], real_data[j], diff_thd)
        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            row = value_format_str.format(
                start + idx,
                str(expect_data[j]),
                str(real_data[j]),
                diff_abs,
                str(diff_rate)
            )
        else:
            diff_abs = abs(np.float64(expect_data[j]) - np.float64(real_data[j]))
            row = value_format_float.format(
                start + idx,
                np.float64(expect_data[j]),
                np.float64(real_data[j]),
                diff_abs,
                diff_rate
            )
        logger.debug(row)


    logger.debug("-" * total_width)
    header = header_format.format("index", "ground_truth", "real", "absolute error", "relative error")
    logger.debug(header)
    logger.debug("-" * total_width)

    split_count = int(end - start)
    if split_count <= 10:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(3):
            display_inner(i)
        ellipsis_row = ellipsis_format.format("  ...  ", "...", "...", "...", "...")
        logger.debug(ellipsis_row)
        for i in range(split_count - 3 + 1, split_count + 1):
            display_inner(i)


def data_compare(npu_output, ground_truth_output, diff_thd=0.001, pct_thd=0.999, max_diff_hd=0.1):
    if isinstance(npu_output, list):
        logger.warning("\033[93m[> Warning <]\033[0m",
            "The first value passed to data_compare is a list, "
            "and the tool will default to comparing the 0th value for you.")
        npu_output = npu_output[0]
    if isinstance(ground_truth_output, list):
        logger.warning("\033[93m[> Warning <]\033[0m",
            "The first value passed to data_compare is a list, "
            "and the tool will default to comparing the 0th value for you.")
        ground_truth_output = ground_truth_output[0]
    if not isinstance(npu_output, np.ndarray):
        npu_output = np.array(npu_output.to(torch.float32))
    if not isinstance(ground_truth_output, np.ndarray):
        ground_truth_output = np.array(ground_truth_output.to(torch.float32))
    if npu_output.dtype == "|V2":
        import bfloat16ext
        npu_output.dtype = "bfloat16"
    max_error_idx = 10000000
    real_data = npu_output.flatten()
    data_compe = ground_truth_output.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        logger.warning('The npu_output is [],and it is same as bm_output, the result of data_compare is \"Pass\"')
        return "Pass", 100.0, 0
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"
    if real_data.size != data_compe.size:
        logger.error("\033[31m[> error <]\033[0m", "Precision comparison failed.")
        logger.error("\033[31m[> error <]\033[0m",
            f"The size of first parameters in data_compare is {real_data.size}, "
            f"and the size of second parameters is {data_compe.size}. The sizes of comparisons are not the same.")
        return result, 0.0, max_error

    overflows_count = data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
    if overflows_count > 0:
        logger.warning('Overflow, size:%s, benchmark_output:%s, %s' %
            (overflows_count, data_compe[np.isinf(data_compe)][0:10], data_compe[np.isnan(data_compe)][0:10]))

    split_count = int(end - start + 1) if end != start else 1
    try:
        diff_abs = np.abs(np.subtract(real_data.astype(np.float32), data_compe.astype(np.float32)))
    except MemoryError:
        return result, 0.0, max_error
    diff_index = np.where(diff_abs > 0)
    rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32),
        data_compe[diff_index].astype(np.float32), diff_thd)
    err_diff = rdiff[rdiff > diff_thd]
    fulfill_percent = float(split_count - err_diff.size) / float(split_count) * 100.0

    display_output(real_data, data_compe, start, end, diff_thd)
    pct_thd = pct_thd * 100.0
    result = "success" if (fulfill_percent >= pct_thd) else "failed"
    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error >= max_diff_hd:
            result = "failed"
    logger.debug('------------------------------------------------------------------------')
    logger.debug('DiffThd  \t PctThd   \t PctRlt   \t Result')
    logger.debug('%.4f  \t %.2f%%   \t %.6f%%   \t %s' %
              (diff_thd, pct_thd, fulfill_percent, result))
    logger.debug('------------------------------------------------------------------------')
    if len(err_diff) > 0:
        logger.debug('Maximum relative error: %s. Maximum relative error threshold: %s.' %(max_error, max_diff_hd))
        logger.debug('------------------------------------------------------------------------')

    return result, fulfill_percent, max_error