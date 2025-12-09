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

from abc import ABC, abstractmethod
from .pattern_match_pass import PatternMatchPass


class PatternBase(ABC):
    @staticmethod
    @abstractmethod
    def inputs():
        raise NotImplementedError(
            f"Pass must implement the 'pattern' method!"
        )

    @staticmethod
    @abstractmethod
    def pattern(*args, **kwargs):
        raise NotImplementedError(
            f"Pass must implement the 'pattern' method!"
        )

    @staticmethod
    @abstractmethod
    def replacement(*args, **kwargs):
        raise NotImplementedError(
            f"Pass must implement the 'replacement' method!"
        )


patterns = PatternMatchPass()


def register_pattern_to_pass(cls: PatternBase):
    name = cls.__name__
    patterns.register_pattern(name, cls.pattern, cls.replacement, cls.inputs())