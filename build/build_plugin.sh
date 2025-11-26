#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

set -e

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

set +e
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
set -e

USER_ABI_VERSION_RAW=$(python3 -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "Error: Failed to retrieve PyTorch ABI version!"
    echo "Possible reasons: PyTorch is not installed, or the version is too old (missing the _GLIBCXX_USE_CXX11_ABI attribute)."
    exit 1
fi
export USER_ABI_VERSION=$(echo "$USER_ABI_VERSION_RAW" | tr -d '[:space:]')

rm -rf build
mkdir -p build
cmake -B build ../csrc
cmake --build build -j