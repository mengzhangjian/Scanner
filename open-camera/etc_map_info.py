#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename    :etc_map_info.py
@brief       :
@time        :2021/01/31 18:54:11
@author      :hscoder
@versions    :1.0
@email       :hscoder@163.com
@usage       :
Copyright (c) 2020. All rights reserved.Created by hanshuo
'''

import os
import traceback

plate_map_info = {'bea9': '京', 'bdf2': '津', 'bba6': '沪', 'd3e5': '渝', 'bcbd': '冀', 'd4a5': '豫', 'd4c6': '云', 'c1c9': '辽', 'bada': '黑', 'cfe6': '湘', 'cdee': '皖', 'c2b3': '鲁', 'd0c2': '新', 'cbd5': '苏', 'd5e3': '浙', 'b8d3': '赣', 'b6f5': '鄂', 'b9f0': '桂', 'b8ca': '甘', 'bdfa': '晋', 'c3c9': '蒙', 'c9c2': '陕', 'bcaa': '吉', 'c3f6': '闽', 'b9f3': '贵', 'd4c1': '粤', 'c7e0': '青', 'b2d8': '藏', 'b4a8': '川', 'c4fe': '宁', 'c7ed': '琼'}


def convert_bytes_chinese(etc_str):
    try:
        etc_str = etc_str.replace('\\x' , '')
        etc_str = etc_str.replace(etc_str[:4] , plate_map_info[etc_str[:4]])
    except BaseException as e:
        etc_str = "000000000"
    return etc_str

def parse_ser_data(item_data):    
    sub_str = item_data.split('&')
    gps = sub_str[0]
    ground = sub_str[1]
    plate_number = convert_bytes_chinese(sub_str[2])   
    return gps , ground , plate_number
