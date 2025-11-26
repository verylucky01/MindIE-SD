# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import sys
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check
sys.path.append(os.path.dirname(__file__))
from flash_attention_fwd import flash_attention_compute


@register_op_compute("flash_attention_tik")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def flash_attention_tik(q, k, v, y, kernel_name="flash_attention_tik"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """
    flash_attention_compute(q, k, v, y, kernel_name)
