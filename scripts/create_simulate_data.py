#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename    :create_simulate_data.py
@brief       :生成数据测试etc逻辑
@time        :2020/09/13 00:40:50
@author      :hscoder
@versions    :1.0
@email       :hscoder@163.com
@usage       :
'''

import os
from glob import glob
import argparse
from shutil import rmtree , copyfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i' , '--image' , help = 'image dir' , default="/workspace/mnt/image/2020-05-05-2")
    parser.add_argument('-o' , '--output' , help= 'save dir' , default="/workspace/mnt/image/2020-05-05-2-new")
    args = vars(parser.parse_args())

    if os.path.exists(args['output']):
        rmtree(args['output'])
        os.mkdir(args['output'])
    else:
        os.mkdir(args['output'])

    img_lsts = glob(args['image'] + "/*.jpg")
    for name in img_lsts:
        base_name = os.path.basename(name)[:-4]
        new_name = base_name + "_000000000.jpg"
        new_name = os.path.join(args['output'] , new_name)
        # import pdb; pdb.set_trace()
        copyfile(name , new_name)

    print('finished')




