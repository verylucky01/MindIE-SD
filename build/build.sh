#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

set -e
BUILD_DIR=$(dirname $(readlink -f $0))
PROJ_ROOT_DIR=${BUILD_DIR}/..
export OUTPUT_DIR=$PROJ_ROOT_DIR/output

if [ ! -d "$OUTPUT_DIR" ];then
    mkdir -p $OUTPUT_DIR
fi

AFT_PATH=${PROJ_ROOT_DIR}
chmod a-w $BUILD_DIR/*

rm -rf ${PROJ_ROOT_DIR}/dist/*
cd ${PROJ_ROOT_DIR}
MindIESDVersion="1.0.RC1"
if [ ! -f "${PROJ_ROOT_DIR}"/../CI/config/version.ini ]; then
    echo "version.ini is not exsited !"
else
    MindIESDVersion=$(cat ${PROJ_ROOT_DIR}/../CI/config/version.ini | grep "PackageName" | cut -d "=" -f 2)
fi
MindIESDVersion=$(echo $MindIESDVersion | sed -E 's/([0-9]+)\.([0-9]+)\.RC([0-9]+)\.([0-9]+)/\1.\2rc\3.post\4/')
MindIESDWheelVersion=$(echo $MindIESDVersion | sed -s 's!.T!.alpha!')
MindIESDVersion=$(echo $MindIESDVersion | sed -s 's!.T!+t!')
echo "MindIESDVersion $MindIESDVersion"
echo "MindIESDWheelVersion $MindIESDWheelVersion"
python3 ${PROJ_ROOT_DIR}/setup.py --setup_cmd='bdist_wheel' --version=${MindIESDWheelVersion}

# "aarch64" / "x86_64"
ARCH=$(uname -m)
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "x86_64" ]]; then
    echo "It is not system of aarch64 or x86_64"
    exit 1
fi

PYTHON_VERSION=""
if command -v python3 &> /dev/null; then
    version=$(python3 --version | awk '{print$2}')
    major=$(echo $version | cut -d '.' -f 1)
    minor=$(echo $version | cut -d '.' -f 2)
    PYTHON_VERSION="py${major}${minor}"
    echo "python version is: $PYTHON_VERSION"
else
    echo "cannot get python version"
    exit 1
fi

rm -rf $OUTPUT_DIR/*
mkdir -p $OUTPUT_DIR/Ascend-mindie-sd_${MindIESDVersion}_${PYTHON_VERSION}_linux_${ARCH}
cp ${PROJ_ROOT_DIR}/dist/mindie*.whl ${OUTPUT_DIR}/Ascend-mindie-sd_${MindIESDVersion}_${PYTHON_VERSION}_linux_${ARCH}
cp ${PROJ_ROOT_DIR}/requirements.txt ${OUTPUT_DIR}/Ascend-mindie-sd_${MindIESDVersion}_${PYTHON_VERSION}_linux_${ARCH}

if [ -n "$AFT_PATH" ] && [ -d "$AFT_PATH" ]; then
    export RELEASE_TMP_DIR=${OUTPUT_DIR}/Ascend-mindie-sd_${MindIESDVersion}_${PYTHON_VERSION}_linux_${ARCH}
    source ${AFT_PATH}/build/build_ops.sh ${AFT_PATH}/build
    SET_ENV_PATH=${OUTPUT_DIR}/Ascend-mindie-sd_${MindIESDVersion}_${PYTHON_VERSION}_linux_${ARCH}/set_env.sh
    touch ${SET_ENV_PATH}
    echo "path=\${BASH_SOURCE[0]}" >> ${SET_ENV_PATH}
    echo "SD_OPS_HOME=\$(cd \$(dirname \$path); pwd )" >> ${SET_ENV_PATH}
    echo "export ASCEND_CUSTOM_OPP_PATH=\${SD_OPS_HOME}/vendors/customize:\${ASCEND_CUSTOM_OPP_PATH}" >> ${SET_ENV_PATH}
    echo "export ASCEND_CUSTOM_OPP_PATH=\${SD_OPS_HOME}/vendors/aie_ascendc:\${ASCEND_CUSTOM_OPP_PATH}" >> ${SET_ENV_PATH}
elif [ -n "$AFT_PATH" ]; then
    echo "Waring: The path of ascend-faster-transformer $AFT_PATH does not exist."
fi

cd $OUTPUT_DIR
tar_package_name="Ascend-mindie-sd_${MindIESDVersion}_${PYTHON_VERSION}_linux_${ARCH}.tar.gz"
tar czf $tar_package_name ./Ascend-mindie-sd_${MindIESDVersion}_${PYTHON_VERSION}_linux_${ARCH} --owner=0 --group=0
rm -rf ./Ascend-mindie-sd_${MindIESDVersion}_${PYTHON_VERSION}_linux_${ARCH}
