#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename    :utils.py
@brief       :
@time        :2021/01/31 20:54:23
@author      :hscoder
@versions    :1.0
@email       :hscoder@163.com
@usage       :
Copyright (c) 2020. All rights reserved.Created by hanshuo
'''

import difflib

#判断相似度的方法，用到了difflib库
def get_equal_rate_1(str1, str2):
   return difflib.SequenceMatcher(None, str1, str2).quick_ratio()