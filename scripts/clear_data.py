#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename    :clear_data.py
@brief       :将乱序数据按照时间戳重新编号
@time        :2020/05/16 10:15:53
@author      :hscoder
@versions    :1.0
@email       :hscoder@163.com
@usage       :
'''

from imutils.paths import list_images
import os
import argparse
import datetime
import shutil

def cmp_stamp(name_path):
    base_name = name_path.split('/')[-1]
    return float(base_name.split('-')[-1].split('_')[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i' , '--image' , default='/data/image/2020-09-19.bak-1')
    parser.add_argument('-s' , '--save'  , default='/data/image/2020-09-19.bak-1-new')

    args =vars(parser.parse_args())
    img_list = list(list_images(args['image']))

    if os.path.exists(args['save']):
        shutil.rmtree(args['save'])
    os.mkdir(args['save'])    
    
    img_list.sort(key = cmp_stamp)

    new_images = []
    for i , img_name in enumerate(img_list):
        base_name = os.path.basename(img_name)
        new_name = str(i) + "-" + base_name.split('-')[-1]
        print(new_name)

        full_src_name = os.path.join(args['image'] , base_name)
        full_dst_name = os.path.join(args['save'] , new_name)

        print(full_src_name)
        print(full_dst_name)
        print('--------------')

        shutil.copyfile(full_src_name , full_dst_name)
