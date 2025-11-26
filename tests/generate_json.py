#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import argparse
import os
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path",
                        default="./t2v_sora.txt", type=str,
                        help='path of prompts')
    parser.add_argument("--input_path",
                        default="./t2v_sora/",
                        type=str, help='input file dirtory path')
    parser.add_argument("--output_path",
                        default="./test.json",
                        type=str, help='output json  path')
    args = parser.parse_args()

    with open(args.prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    vidio_name = sorted(os.listdir(args.input_path))
    res = {}
    for i, _ in enumerate(prompts):
        res[os.path.join(args.input_path, vidio_name[i])] = prompts[i]
    fd = os.open(args.output_path, os.O_WRONLY | os.O_CREAT, 0o644)
    with open(fd, 'w',) as f:
        json.dump(res, f)


