#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename    :debugger.py
@brief       :包含调试时的一些辅助函数
@time        :2020/04/20 20:48:37
@author      :hscoder
@versions    :1.0
@email       :hscoder@163.com
@usage       :
'''

import cv2
import os
import numpy as numpu
from copy import deepcopy


def draw_ssd(rets, cv_img):
    '''
    绘制ssd的检测结果
    '''
    assert len(rets) != 0

    h , w = cv_img.shape[:2]
    tmp_img = deepcopy(cv_img)

    for ret in rets:
        name, conf, [x1, y1, x2, y2, area] = ret
        if x2 + 20 >= w or y2 + 20 >= h:
            continue

        if name == 'car' and area <= 60000:
            continue
            
        cv2.rectangle(tmp_img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        cv2.putText(tmp_img, str(name) + ":" + str(round(conf , 2)), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , thickness=2)

    return tmp_img


def draw_plate(plate_res, name , start_x, start_y):
     '''
     plate_res:[plcode, score, [left, top, right, bottom], [name, score]]
     start_x:car box的左上角x
     start_y:car box的左上角y
     name:image name
     '''
     assert len(plate_res) != 0
     cv_img = cv2.imread(os.path.join("middle_result", name))

     x1 = start_x + plate_res[2][0]
     y1 = start_y + plate_res[2][1]
     x2 = plate_res[2][2] + start_x
     y2 = plate_res[2][3] + start_y

     cv2.rectangle(cv_img, (x1 , y1), (x2 , y2), (0, 255, 0), thickness = 2)
     cv2.putText(cv_img, str("plate"), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , thickness=2)
     return cv_img
