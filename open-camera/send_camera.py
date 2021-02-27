# -*- coding: utf-8 -*-


import pika
import cv2
import time
import os
from multiprocessing import Process, Manager
from redisProcess import RfidRedis
import sys
import traceback
from camera import IPVideoCapture
from shutil import rmtree
from log import get_logger
from etc_map_info import parse_ser_data

import psutil
import requests
import json
from ctypes import c_char_p

time.sleep(10)

rfid_obj = RfidRedis('redis')
rfid_obj.clear()

camera_log_dir = 'camera_log'
if os.path.exists(camera_log_dir):
    rmtree(camera_log_dir)
    os.mkdir(camera_log_dir)
else:
    os.mkdir(camera_log_dir)
my_logger = get_logger(os.path.join(camera_log_dir , 'camera_log.log'))

ser = None
online_model = True
if online_model:
    import serial
    import RPi.GPIO as GPIO
    url_rtsp = 'rtsp://admin:abc12345678@192.168.1.64:554/'  # 根据摄像头设置IP及rtsp端口
    status = os.system('sh /work/start_xhost.sh')
    my_logger.info('open uart port success')
    """open serial port"""

    try:
        ser = serial.Serial("/dev/ttyTHS0", 115200 , timeout = 0 , inter_byte_timeout=0.1)
    except BaseException as e:
        my_logger.info('open serial failture ---{}'.format(e))
        my_logger.info(traceback.format_exc())


    """rabbitmq initialize"""
    try:
        credentials = pika.PlainCredentials('chepai', '123456')
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='vbs-rabbitmq', port=5672, credentials=credentials, heartbeat=0))
    except BaseException as e:
        my_logger.info('rabbitmq init occur problem --- {}'.format(e))
        my_logger.info(traceback.format_exc())
else:
    try:
        url_rtsp_mp4 = '2.mp4'
        credentials = pika.PlainCredentials('chepai', '123456')
        connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='vbs-rabbitmq', port=5672, credentials=credentials, heartbeat=0))
    except BaseException as e:
        my_logger.info('offline rabbitmq init occur problem --- {}'.format(e))
        my_logger.info(traceback.format_exc())

channel = connection.channel()
channel.queue_declare(queue='hello', durable=True)

# Pin Definitions:
led_pin_4 = 23
led_pin_5 =29 

def open_cam_rtsp(uri, width, height, latency):
    gst_str = ("rtspsrc location={} latency={} ! rtph264depay ! h264parse ! omxh264dec ! "
               "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
               "videoconvert ! appsink").format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def ser_line_echo(x, gpshb):
    """x: input list for gps-rfid"""
    if online_model:
        global ser
        if ser.is_open == False:
            ser.open()
        while True:
            try:
                count = ser.inWaiting()
                data = ser.read(size=4095)
                item = str(data, encoding='gb18030')
                
                if item:
                    x.append(item[2:67])
                    gpshb.value = item[2:67].split('&')[0]

            except BaseException as e:
                my_logger.info('read serial occur failture , attemp again')
                my_logger.info(traceback.format_exc())
                my_logger.info('trying processing serial,restarting')
                ser = serial.Serial("/dev/ttyTHS0", 115200 , timeout = 0 , inter_byte_timeout=0.1)
                continue


def show_camera(show_flag=False):
    if show_flag:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        unix_time = time.time()
        dataTime = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(int(unix_time))).split(' ')[0]
        save_path = "/data/video"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        path = os.path.join(save_path, dataTime)
        if not os.path.exists(path):
            os.mkdir(path)
        out = cv2.VideoWriter(os.path.join(path , dataTime + ".mp4") , fourcc, 5.0, (1920, 1080))
        if online_model:
            cap = open_cam_rtsp(url_rtsp, 1920, 1080, 50)
        else:
            cap = cv2.VideoCapture(url_rtsp_mp4)
        ret = True
        while ret:
            ret, frame = cap.read()
            out.write(frame)
    else:
        pass

def blink():
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup([led_pin_4], GPIO.OUT)  # LED pins set as output
    GPIO.output(led_pin_4, GPIO.HIGH)

    while True:
        GPIO.output(led_pin_4, GPIO.LOW)
        time.sleep(0.5)
        GPIO.output(led_pin_4, GPIO.HIGH)
        time.sleep(0.5)

def blink5():
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup([led_pin_5], GPIO.OUT)  # LED pins set as output
    GPIO.output(led_pin_5, GPIO.HIGH)

    for i in range(1000):
        GPIO.output(led_pin_5, GPIO.LOW)
        time.sleep(0.5)
        GPIO.output(led_pin_5, GPIO.HIGH)
        time.sleep(0.5)


def main(x):
    if online_model:
        """x: list for gps_rfid"""
        while True:
            try:
                cap = IPVideoCapture(blocking=True)
                my_logger.info('connect camera successful')
                break
            except:
                my_logger.info('connect camera failed, trying connecting')
                continue
    else:
        cap = cv2.VideoCapture(url_rtsp_mp4)

    save_path = "/data/image"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    index = 0
    frame_index = 0
    fps_integer = 5


    old_rifd = "E000.000000,N00.000000&000000000000&000000000&"
    rfid_dict = {}
    while True:
        try:
            if len(x) > 0:
                gps_rfid = x.pop(0)
            else:
                gps_rfid = "E000.000000,N00.000000&000000000000&000000000&"
            ret, frame = cap.read()
            index += 1
            if index % fps_integer != 0:
                continue
            unix_time = time.time()
            dataTime = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(int(unix_time))).split(' ')[0]
            path = os.path.join(save_path, dataTime)
            if not os.path.exists(path):
                os.mkdir(path)
            
            try:
                my_logger.info('gps_rfid: {} , type: {}'.format(gps_rfid , type(gps_rfid)))                
                gps , ground , plate_number = parse_ser_data(gps_rfid)
                if ground  not in rfid_dict:
                   rfid_dict[ground] = [1, unix_time]
                if (unix_time - rfid_dict[ground][1]) > 50:
                    rfid_dict[ground] =[1, unix_time]
                else:
                    rfid_dict[ground][1] = time.time()
                    rfid_dict[ground][0] += 1
                if rfid_dict[ground][0] > 500:
                    continue
            except BaseException as e:
                my_logger.info('process split gps_rfid occur ---{}'.format(e))
                gps = "E000.000000,N00.000000"
                ground = "000000000000"
                plate_number = "000000000"

            img_name = os.path.join(path, str(
                frame_index) + "-" + str(unix_time) + '_' + str(gps) + '_' + str(ground) + "_" + str(plate_number) + '.jpg')
            cv2.imwrite(img_name, frame)
            channel.basic_publish(
                exchange='', routing_key='hello', body=img_name)
            print("{} send image, gps {} rfid {} etc {}".format(frame_index, gps, ground , plate_number))

            if str(ground) == '000000000000':
                    pass
            else:
                rfid_obj.add(value="{}_{}_{}".format(str(unix_time) , str(gps) , str(ground)))

            if plate_number == '000000000':
                pass
            else:
                rfid_obj.add(key='plate' , value="{}_{}".format(str(unix_time) , str(plate_number)))
                
            frame_index += 1
 
        except BaseException as e:
            my_logger.info('read frame occur --- {}'.format(e))
            my_logger.info(traceback.format_exc())
            if online_model:
                GPIO.cleanup()
            else:
                pass
            continue

# 发送心跳数据
def send_heartbeatgps(gpshb):
    sleeptime = 5
    time.sleep(sleeptime)
    while True:
        try:
            heaturl = "http://39.106.222.9:50030/MchApi/evt/receive"
            heatdict = {}
            heatdict["evt"] = "evt.scanner.keep"
            heatdict["deviceId"] = "Firston-MVB-0035"
            heatdict["longitude"] = float(gpshb.value.split(',')[0][1:])
            heatdict["latitude"] = float(gpshb.value.split(',')[1][1:])
            heatdict["hddSize"] = round(psutil.disk_usage("/work").total/(1024*1024*1024),2)
            heatdict["hddUsed"] = round(psutil.disk_usage("/work").used/(1024*1024*1024),2)
            heatdict["hddAvail"] = round(psutil.disk_usage("/work").free/(1024*1024*1024),2)
            heatdict["version"] = "V1.0"
            heatdict["electric"] = 0
            heatdict["gpuUsed"] = 0.0
            heatdict["dataTime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
            my_logger.info('heartbeatgps: {} '.format(heatdict))
            res = requests.post(heaturl, json = heatdict, timeout = 500)
            resjson = json.loads(res.text)
            if resjson["errorcode"] == "0":
                my_logger.info('send_heartbeatgps res: {} '.format(resjson["errorcode"]))
            elif resjson["errorcode"] == "1":
                my_logger.info('send_heartbeatgps res sleeptime: {} '.format(resjson["message"]))
                sleeptime = int(resjson["message"])
        except Exception as e:
            my_logger.info('send_heartbeatgps: {} '.format(e))
        finally:
            time.sleep(sleeptime)

if __name__ == "__main__":
    manager = Manager()
    my_list = manager.list()
    gpshb = manager.Value(c_char_p, "E000.000000,N00.000000")
    p1 = Process(target=main, args=(my_list,))
    p2 = Process(target=show_camera, args=(False,))
    p3 = Process(target=ser_line_echo, args=(my_list, gpshb,))
    p4 = Process(target=blink)
    p5 = Process(target=send_heartbeatgps, args=(gpshb,))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
