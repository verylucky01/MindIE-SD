#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:

#     http://license.coscl.org.cn/MulanPSL2

# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, 
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import logging
import os
import sys
import argparse
import subprocess
from setuptools import setup, find_packages, Extension

os.environ["SOURCE_DATE_EPOCH"] = "0"


def copy_so_files(src_dir, dest_dir):
    # 确保目标目录存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 构建cp命令
    so_files = [f for f in os.listdir(src_dir) if f.endswith('.so')]
    if not so_files:
        logging.warning(f"No .so files found in {src_dir}")
        return

    for so_file in so_files:
        src_file = os.path.join(src_dir, so_file)
        dest_file = os.path.join(dest_dir, so_file)
        # 使用subprocess调用cp命令
        subprocess.check_call(['/bin/cp', '-v', src_file, dest_file])  # -v选项用于显示详细信息
        logging.info(f"Copied {src_file} to {dest_file}")


build_script_path = os.path.join(os.path.abspath(os.getcwd()), 'build')
subprocess.check_call(['bash', './build_plugin.sh'], cwd=build_script_path)

source_dir = os.path.join(build_script_path, 'build')  # .so文件位于csrc/build/
destination_dir = os.path.join(os.path.abspath(os.getcwd()), 'mindiesd', 'plugin')  # 目标位置为mindiesd/plugin/
copy_so_files(source_dir, destination_dir)

# 创建一个参数解析实例
parser = argparse.ArgumentParser(description="Setup Parameters")
parser.add_argument("--setup_cmd", type=str, default="bdist_wheel")
parser.add_argument("--version", type=str, default="1.0.RC1")

# 开始解析
args, unknown = parser.parse_known_args()
# 把路径参数从系统命令中移除，只保留setup需要的参数
sys.argv = [sys.argv[0], args.setup_cmd] + unknown
logging.info(sys.argv)

mindie_sd_version = args.version

setup(
    name="mindiesd",
    version=mindie_sd_version,
    author="ascend",
    description="build wheel for mindie sd",
    setup_requires=[],
    zip_safe=False,
    python_requires=">=3.10",
    include_package_data=True,
    packages=find_packages(),
    package_data={"": ["*.so"]},
)
