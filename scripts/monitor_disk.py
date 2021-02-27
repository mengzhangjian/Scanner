#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename    :monitor_disk.py
@brief       :监控磁盘空间，当占用超过磁盘空间百分之六十时，按照目录的创建时间自动清理
@time        :2020/05/16 21:43:38
@author      :hscoder
@versions    :1.0
@email       :hscoder@163.com
@usage       :
'''

import os
import shutil
import psutil
from apscheduler.schedulers.blocking import BlockingScheduler
import time
from setproctitle import setproctitle
from glob import glob
import time
import argparse

one_day_ago = 60 * 60 * 24 
def clean_disk(dir_path):
    seiz = psutil.disk_usage("/data")
    print('seiz: ' , seiz.percent)
    if seiz.percent > 60:
        # todo
        print('开始清理磁盘')
        # folders = glob(os.path.join(dir_path , '*'))
        folders = os.listdir(dir_path)
        print(folders)
        

        folders = sorted(folders , key = lambda item : os.path.getctime(os.path.join(dir_path , item)))
        for f in folders:
            full_dir = os.path.join(dir_path , f)
            if os.path.isdir(full_dir):
                timeStamp = os.path.getctime(full_dir)

                if int(time.time() - timeStamp) > one_day_ago:
                    print(f)
                    shutil.rmtree(full_dir)
                    seiz = psutil.disk_usage("/")
                    if seiz.percent < 60:
                        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i' , '--disk' , default='/data/image' , help='监控的目录')
    args = vars(parser.parse_args())
    setproctitle("monitor_disk")
  
    os.chmod(args['disk'], 0o777)

    try:
        print('首次清理')
        clean_disk(args['disk'])
    except BaseException as e:
        print('执行监控磁盘脚本发生异常---{}'.format(e))

    try:
        scheduler = BlockingScheduler()
        print('定时清理')
        scheduler.add_job(clean_disk , args=(args['disk'] ,) , trigger= 'interval' , hours = 1)
        scheduler.start()

    except BaseException as e:
        print('启动监控发生异常---{}'.format(e))


