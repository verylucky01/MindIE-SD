#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
export MINDIE_LOG_TO_STDOUT=true

set -e

if command -v python3 &> /dev/null; then
    python_command=python3
else
    python_command=python
fi

current_directory=$(dirname "$(readlink -f "$0")")

${python_command} -m coverage run --branch --source=../mindiesd ${current_directory}/run.py 2>&1 | tee ${current_directory}/run.log

${python_command} -m coverage report
${python_command} -m coverage xml -o ${current_directory}/coverage.xml

${python_command} ${current_directory}/scripts/unittest_summary.py ${current_directory}/run.log
