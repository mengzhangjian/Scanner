# coding=utf-8
import os
import time
import cv2
import sys
import pika
import uuid
import json
import glob
import pickle
import requests
import logging
import base64
import asyncio
import traceback
from sort import *

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from multiprocessing import JoinableQueue
import multiprocessing

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)

online_model = True
Debug = False if online_model else True

Device_Id = str("01")

# 这里板子上默认是注释掉的，如果有问题请尝试注释
# sess.run(tf.global_variables_initializer())
graph = tf.get_default_graph()

sess2 = tf.Session(config=config)
grah2 = tf.get_default_graph()

set_session(sess)

post_url = "http://xjpark.firston-tech.com:50030/MchApi/evt/receive"  # 218.241.189.6

# A
post_url_A = "http://47.94.83.100:50030/MchApi/evt/receive"
# B
post_url_B = "http://218.241.189.5:50030/MchApi/evt/receive"

headers = {'Content-type': 'application/json'}


car_detection_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../car_detection_6")
sys.path.append(car_detection_path)
print(car_detection_path)

car_plate_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../car_plate_2")
sys.path.append(car_plate_path)

plate_color_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../plate_color_3")
sys.path.append(plate_color_path)

car_model_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../car_model_1")
sys.path.append(car_model_path)

api_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../api")
sys.path.append(api_path)

redis_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../redisServer")
sys.path.append(redis_path)
logger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../api")
sys.path.append(logger_path)

from parse_config import ConfigParser
from debugger import draw_ssd, draw_plate
from server_launcher import detection_func as ssd_cheliang_detect
from webplate import car_plate_detect
from webapp import color_detect
from webCarModelapp import car_model_detect
from redisProcess import RfidRedis
from utils_etc import get_equal_rate_1
from os.path import exists, basename

my_log = ConfigParser('log/log')
result_log =  ConfigParser('log/result')

my_logger = my_log.get_logger("plate_log", 1)
result_logger =  result_log.get_logger("result_log", 1)


loop = asyncio.get_event_loop()

"""rabbitmq initialize"""
if online_model:
    # rabbitMQ
    credentials = pika.PlainCredentials('chepai', '123456')
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='vbs-rabbitmq', port=5672, credentials=credentials, heartbeat=0))

    # redis
    redis_object = RfidRedis(host='redis')
else:
    # rabbitMQ
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost', port=5672, heartbeat=0))

    # redis
    redis_object = RfidRedis()

channel = connection.channel()
channel.queue_declare(queue='hello')
channel.queue_declare(queue='send')

tracker = Sort()

history = []
history_id = []
car_info = {}

name_reverse = {"car": 0, "plate": 1, "ground": 2}

# 记录发送给后台的图片
saveResult = "result"

# 记录中间结果图片
debug_middle_dir = "middle_result"

# 记录车辆检测图片

debug_car_dir = "car_result"

all_dir = "have_with_no"

W, H = None, None

pre_track_id = 0
cur_track_id = 0

pre_rfid = '000000000000'
cur_rfid = '000000000000'

empty_carport_queue = JoinableQueue()
result_carport_queue = JoinableQueue()

if Debug:
    if exists(saveResult):
        rmtree(saveResult)
    os.mkdir(saveResult)

    if exists(debug_middle_dir):
        rmtree(debug_middle_dir)
    os.mkdir(debug_middle_dir)

    if exists(debug_car_dir):
        rmtree(debug_car_dir)
    os.mkdir(debug_car_dir)

    if exists(all_dir):
        rmtree(all_dir)
    os.mkdir(all_dir)


class Car:
    """Car class"""

    def __init__(self, car_id=None, plate=[None, None], ground=None, timeStamp=None, full_name=None,
                 gps=None, rfid=None, carBox=None,
                 ground_id=None, name=None, shape=None, Platecolor=None, ground_num=0, send_server=False):
        """plate = [result, probability]"""
        self.timeStamp = timeStamp
        self.fullName = full_name
        self.gps = gps
        self.rfid = rfid
        self.car_id = car_id
        self.plate = plate  # [code, score, [1,2,3,4]]
        self.ground = ground
        self.carBox = carBox
        self.ground_id = ground_id
        self.name = name
        self.shape = shape
        self.plate_color = Platecolor
        self.ground_num = ground_num
        self.send_server = send_server

    def trigger_full(self):
        if self.plate[0] is not None and self.ground is not None:
            return True
        return False

    def trigger_ground_alone(self):
        if self.ground is not None and self.plate[0] is None:
            return True

        return False

    def trigger_plate_alone(self):
        if self.plate[0] is not None and self.ground is None:
            return True
        return False

    def judge_empty(self):
        if self.car_id is None and self.ground_id is not None \
                and self.ground is not None:
            return True
        return False


def compute_iou(reframe, gtframe, reverse=False):
    """
    calc IOU
    :param reframe:
    :param gtframe:
    :return: ratio
    """

    cx1, cy1, cx2, cy2 = reframe
    gx1, gy1, gx2, gy2 = gtframe

    carea = (cx2 - cx1) * (cy2 - cy1)
    garea = (gx2 - gx1) * (gy2 - gy1)

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    area = w * h
    if reverse:
        ratio = area / carea if area > 0 else 0
        return ratio

    ratio = area / (carea + garea - area) if area > 0 else 0

    return ratio


def expand_bowei_iou_with_car(frame, car_box, ground_box):
    """
    :param frame:
    :param carBox:
    :return:
    """
    h, w = frame.shape[:2]
    left, top, right, bottom = ground_box

    car_iou_top = top - h // 2
    if car_iou_top < 0:
        car_iou_top = 0
    car_iou_right = right + w // 2
    if car_iou_right > w:
        car_iou_right = w - 10

    if compute_iou([left, car_iou_top, car_iou_right, bottom], car_box) > 0.1:
        return True

    return False


def track_detect(boxes):
    """format tracking car result"""
    dets = []
    for b in boxes:
        cat = b[0]
        if cat in name_reverse:
            cls = name_reverse[cat]
            score = b[1]
            left, top, right, bottom = b[2][:4]
            dets.append([left, top, right, bottom, float(score), cls])
    return dets


def depart_class(box):
    """depart object to its list
        return car,plate,ground
    """

    global W, H
    car = []
    plate = []
    ground = []
    box_ground = []
    for det in box:
        cat, conf, [x1, y1, x2, y2, area] = det[0], det[1], det[2]

        if x2 + 10 >= W or y2 + 10 >= H:
            continue

        if cat == "car" and area >= 90000 and (x1 + x2) >= W / 2 and y2 >= H / 2 and conf >= 0.9:
            car.append(det)
        elif cat == "ground":
            ground.append(det)
        else:
            plate.append(det)

    if len(car) > 0:
        # 过滤掉小目标，只选择前2个车
        car = sorted(car, key=lambda x: x[2][4], reverse=True)
        car = car[:2] if len(car) >= 2 else car
    if len(ground) > 0:
        # 选择当前最大目标
        ground = sorted(ground, key=lambda x: x[2][4], reverse=True)[0]
        box = ground[2]
        x1, y1, x2, y2, _ = box
        center_x = x1 + (x2 - x1) // 2
        center_y = y1 + (y2 - y1) // 2
        box_ground.append([center_x, center_y, x1, y1, x2, y2])

    return car, box_ground


def plate_in_car(plate, car):
    """judge plate Box if in car Box
    judge plate box in car box and plate y2 coordiante > car center height
       input: plate Box, Car Box
       return true or false
    """
    cx1, cy1, cx2, cy2 = plate
    gx1, gy1, gx2, gy2 = car

    car_height = gy2 - gy1

    car_center_height = gy1 + car_height // 2

    j1 = True if (gy2 - cy2) >= 0 else False
    j2 = True if (gx2 - cx2) >= 0 else False
    j3 = True if (cx1 - gx1) >= 0 else False
    j4 = True if (cy1 - gy1) >= 0 else False
    j5 = True if (cy2 > car_center_height) else False
    res = True if (j1 and j2 and j3 and j4 and j5) else False

    return res


def zoom_ground_image(img, ground):
    """ expand ground area
        return ground
    """
    h, w = img.shape[:2]
    gx1, gy1, gx2, gy2 = ground[:4]
    scale_h = int(h * 0.015)
    scale_w = int(w * 0.015)
    new_x1 = gx1 - scale_w
    new_y1 = gy1 - scale_h
    new_x2 = gx2 + scale_w
    new_y2 = gy2 + scale_h

    new_x1 = gx1 if new_x1 < 0 else new_x1
    new_y1 = gy1 if new_y1 < 0 else new_y1
    new_x2 = gx2 if new_x2 > w else new_x2
    new_y2 = gy2 if new_y2 > h else new_y2

    return [new_x1, new_y1, new_x2, new_y2]


def recognize_plate_color(img):
    """
    plate color recognition
    :param img:
    :return: [color, score]
    """
    # number = {"blue": 0, "green": 1, "yellow": 2}
    number = {u"蓝牌": 0, u"黄牌": 2, u"新能源": 1, u"白色": 3, u"黑色": 4}
    if online_model:
        with grah2.as_default():
            set_session(sess2)
            result = color_detect(img)["results"]
    else:
        result = color_detect(img)["results"]

    my_logger.info('plate color: {}'.format(result))
    if len(result) > 0:
        color = result[0]["name"]
        score = result[0]["score"]
        n = number[color]
        return [n, score]

    return [None, 0.0]


def recognize_plate(img, start_x=0, start_y=0, name=None):
    """plate recognation
    return [plcode, score, [left, top, right, bottom], [name, score]]
    """

    my_logger.info('recognize plate')

    with graph.as_default():
        set_session(sess)
        start_time = time.time()
        rs = car_plate_detect(img)
        my_logger.info("plate detect cost time: {}".format(time.time() - start_time))

    result = rs["results"]
    if len(result) > 0:
        plcode = result[0]["name"]
        score = result[0]["score"]
        left = result[0]["location"]["left"]
        top = result[0]["location"]["top"]
        right = result[0]["location"]["right"]
        bottom = result[0]["location"]["bottom"]

        plate = img[top: bottom, left: right]

        start_time = time.time()
        color = recognize_plate_color(plate)
        my_logger.info('plate color recognize cost time: {}'.format(time.time() - start_time))

        if score > 0.8:
            if Debug:
                res_plate = [plcode, score, [left, top, right, bottom], color]
                my_logger.info('draw plate: {}'.format(name))
                my_logger.info('res_plate: {}'.format(res_plate))

            return [plcode, score, [left, top, right, bottom], color]
        else:
            my_logger.info('车牌识别的置信度小于阈值---{}'.format(name))
        return [None, 0.0, None, None]

    else:
        my_logger.info("{} 没有识别出车牌或者该车没有挂车牌".format(name))
        return [None, 0.0, None, None]


def recognize_car_shape(img):
    """
    car model recognition
    :param img:
    :return: [shape, score]
    """
    shape_result = car_model_detect(img)['results']
    if len(shape_result) > 0:
        shape = shape_result[0]["name"]
        score = shape_result[0]["score"]
        return [shape, score]

    return [None, 0.0]


def recognize_ground_ssd(img, coordinate):
    """
    part for recognize BoWeiHao
    #ground = zoom_ground_image(img, coordinate)
    # new_img = imutils.rotate_bound(img[ground[1]: ground[3], ground[0]: ground[2]], -70)
    :param img:
    :param coordinate:
    :return:
    """

    my_logger.info('step into recognize_ground_ssd')

    ground_code = None
    gx1, gy1, gx2, gy2 = coordinate[:4]

    try:
        result = yolo_bowei_detect(img[gy1: gy2, gx1: gx2])
        my_logger.info('yolo bowei detect: {}'.format(result))
        stats = result["results"]["state"]
        if stats == 1:
            ground_code = result["results"]["code"]
            score = result["results"]["score"]

            my_logger.info('ground code: {}'.format(ground_code))
            return ground_code, score
        else:
            return None, 0.0
    except BaseException as e:
        my_logger.error('yolo bowei detect occur exception---{}'.format(e))
        my_logger.error(traceback.format_exc())
        return None, 0.0


def detect_car(img, name=None):
    """
    Detection for detect car, plate and ground
    :param img:
    :return:
    """
    results = []
    my_logger.info('step into detect car function')
    results = []
    res = ssd_cheliang_detect(img)
    result = res["results"]
    if len(result) > 0:
        for res in result:
            name = res["name"]
            conf = res["score"]
            left = int(res["location"]["left"])
            top = int(res["location"]["top"])
            right = int(res["location"]["right"])
            bottom = int(res["location"]["bottom"])
            area = res["location"]["area"]
            results.append([name, conf, [left, top, right, bottom, area]])

    return results


async def main_request(frame, carbox, name=None):
    """
    :param plate_img:
    :param car_img:
    :return: 车牌结果 ， 车类型
    """
    x1, y1, x2, y2 = carbox
    car_img = frame[y1:y2, x1:x2, :]

    # create a Coroutine for plate recognize
    plate_recognize = loop.run_in_executor(
        None, recognize_plate, car_img, x1, y1, name)  # plate_img

    # create a Coroutine for car shape
    car_shape = loop.run_in_executor(None, recognize_car_shape, car_img)

    try:
        res1 = await plate_recognize
        res2 = await car_shape
    except Exception as e:
        # print(e)
        my_logger.error('execute main request occur error---{}'.format(e))
        my_logger.error(traceback.format_exc())
        return None, 401

    return (res1, res2), 200


def plate_shape_object(frame, car_box, name=None):
    """
    :param frame:
    :param plate_box:
    :param car_box:
    :return:
    """
    if name:
        my_logger.info(
            'car plate shape recognize , current image name: {}'.format(name))
    else:
        my_logger.info('car plate shape recognize')

    # l2, t2, r2, b2 = car_box
    # car_img = frame[t2: b2, l2: r2]
    res, stats_code = loop.run_until_complete(
        main_request(frame, car_box, name))
    if stats_code == 200:
        res1, res2 = res

        return res1, res2

    return [None, 0.0, None, None], [None, 0.0]


def dist_norm(c, box_ground):
    """lambda for car and ground"""
    x = c.carBox[0] + (c.carBox[2] - c.carBox[0]) / 2
    y = c.carBox[3]
    val = np.linalg.norm(np.array(box_ground[0][:2]) - np.array([x, y]))
    return val


def compare_ground_score(lx, ly):
    """
    :param lx: current: [[name, score], [name, score]]
    :param ly: last: [[name, score], [name, score]]
    :return: [[name, score], [name, score]]
    """
    for idx, item in enumerate(lx):
        if item[1] < ly[idx][1]:
            item[0] = ly[idx][0]


def contact_str(ground):
    """
    :param ground:
    :return:
    """
    gd_str = ''
    for item in ground:
        gd_str += item[0]

    return gd_str


def test_main(img_path):
    global W, H
    global pre_track_id
    global cur_track_id
    global history
    global cur_rfid
    global pre_rfid

    print('process {}'.format(img_path))
    pre_track_id = cur_track_id
    if not exists(img_path):
        my_logger.error('{} is not exist'.format(img_path))
        return

    frame = cv2.imread(img_path)
    car_object = []

    jpg_name = img_path.split('/')[-1]
    split_str = jpg_name[:-4].split('_')
    id = split_str[0].split('-')[0]
    unix_time = split_str[0].split('-')[1]
    gps = split_str[1]
    rfid = split_str[2]

    pre_rfid = cur_rfid
    if rfid == '000000000000':
        rfid = None
    else:
        cur_rfid = rfid
        if cur_rfid != pre_rfid:
            my_logger.info('{} image path: {} empty chewei'.format(id, basename(jpg_name)))
            result = post_empty_carport_result(frame, cur_rfid, gps, unix_time)

            if result['empty'] == 1 and result['berthId'] != '000000000000':
                empty_carport_queue.put(result)

    my_logger.info('{} image path: {}'.format(id, basename(jpg_name)))
    my_logger.info('{} rfid:       {}'.format(id, rfid))

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    draw_frame = None
    try:
        start_time = time.time()
        det_result = detect_car(frame)
        my_logger.info("detect car cost time: {}".format(time.time() - start_time))
        if len(det_result) == 0:
            my_logger.info('{} do not detect car'.format(id))
            return
        if Debug:
            my_logger.info('{} 绘制ssd检测结果并落盘'.format(id))
            draw_frame = draw_ssd(det_result, frame)

    except BaseException as e:
        my_logger.error(
            '{} detectt car occur exception---{}'.format(id, e))
        return

    my_logger.info('{} 分离car和泊位号'.format(id))
    car, ground = depart_class(det_result)
    if len(car) == 0:
        my_logger.info('{} 没有检测到符合要求的车辆'.format(id))
        return

    my_logger.info('{} 开始跟踪'.format(id))
    dets = track_detect(car)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    if len(tracks) > 0:
        for box in tracks:
            # x , y , x1 , y1 = int(box[0]) , int(box[1]) , int(box[2]) , int(box[3])
            [x, y, x1, y1] = list(map(int, box[:4]))
            # box[4]为识别ID，BOX[5]为类别
            if box[5] == 0:
                t = Car(car_id=int(box[4]), timeStamp=unix_time, full_name=img_path, gps=gps, rfid=rfid, carBox=[
                    abs(x), abs(y), x1, y1], name="car", ground_num=len(ground))

                if Debug:
                    cv2.putText(draw_frame, str(cur_track_id), ((x + x1) // 2, (y + y1) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 1)

                if int(box[4]) in car_info and car_info[int(box[4])].send_server:
                    my_logger.info('已经发送给后台，继续处理下一张图')
                    break
                car_object.append(t)

                if int(box[4]) in car_info:
                    if car_info[int(box[4])].plate[0] is None:
                        car_info[int(box[4])] = t

                else:
                    car_info[int(box[4])] = t

                my_logger.info('{} track car id: {}'.format(id, int(box[4])))
                cur_track_id = int(box[4])
    else:
        my_logger.info("{} 跟踪丢失".format(id))
        cur_track_id = 0
        if Debug:
            cv2.putText(draw_frame, str(cur_track_id), (960, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)

    cv2.imwrite(os.path.join(debug_middle_dir, jpg_name), draw_frame)
    # 车牌绑定
    if len(car_object) > 0:
        my_logger.info('start car plate bind')
        for c in car_object:
            try:
                # r1 : 车牌识别接结果 [plate.number , conf , [x1 ,y1 ,x2 , y2] , [color , conf]]
                # r2 : 车的类型
                # 异步处理

                start_time = time.time()
                if Debug:
                    x1, y1, x2, y2 = c.carBox
                    cv2.imwrite(os.path.join(debug_car_dir,
                                             jpg_name), frame[y1:y2, x1:x2, :])
                    r1, r2 = plate_shape_object(frame, c.carBox, jpg_name)
                else:
                    r1, r2 = plate_shape_object(frame, c.carBox)
                my_logger.info("recognize car shape cost time: {}".format(time.time() - start_time))
            except BaseException as e:
                my_logger.error('car plate shape recognize occur exception---{}'.format(e))
                my_logger.error(traceback.format_exc())
                continue

            c.plate = r1
            c.plate_color = r1[3]
            c.shape = r2

            car_ob = car_info[c.car_id]
            # if plate result is not None, compare with the conf , bring one whose conf is greater
            if r1[1] > car_ob.plate[1]:
                car_info[c.car_id] = c

    my_logger.info('len of car_object: {}'.format(len(car_object)))
    if len(history):
        for last in history:
            for current in car_object:
                if last.car_id is not None and current.car_id is not None:
                    if current.car_id == last.car_id:
                        if current.plate[0] is not None and last.plate[0] is not None:
                            if last.plate[1] > current.plate[1]:
                                current.plate = last.plate
                        if last.plate[0] is None and current.plate[0] is not None:
                            last.plate = current.plate
                        if last.ground is not None and current.ground is not None:
                            if last.ground[1] < current.ground[1]:
                                last.ground = current.ground
                        if last.ground is None and current.ground is not None:
                            last.ground = current.ground
                        if last.rfid is None and current.rfid is not None:
                            last.rfid = current.rfid

    my_logger.info('before history: {}'.format(len(history)))
    for x in car_object:
        history.append(x)
        if x.car_id not in history_id:
            # history.append(x)
            history_id.append(x.car_id)
        else:
            pass
    my_logger.info('history_id: {}'.format(history_id))

    my_logger.info('print length of history: {}'.format(len(history)))
    if len(history) >= 2 and cur_track_id != pre_track_id:
        car = history.pop(0)
        history = []

        car_id = car.car_id
        car_ob = car_info[car_id]
        my_logger.info('send_server is {}'.format(car_ob.send_server))
        if car_ob.send_server:
            return

        saveframe = cv2.imread(car_ob.fullName)
        result = post_result(car_ob, saveframe, car_ob.fullName)

        if len(result) > 0:
            try:
                # result_carport_queue.put(result)

                car_ob.send_server = True
                car_info[car_id] = car_ob
                my_logger.info('after send_server: {}'.format(car_ob.send_server))

                result_logger.info('--- detect reuslt ---')

                result_logger.info('current image name:       {}'.format(car_ob.fullName))
                result_logger.info(
                    'current car plate number: {}'.format(result['plateNumber']))
                result_logger.info(
                    'current berthId number  : {}'.format(result['berthId']))
                result_logger.info('current rfidInfo:         {}'.format(result['rfidInfo']))
                result_logger.info(
                    'current car corlor      : {}'.format(result['plateColor']))
                result_logger.info(
                    'current car shape       : {}'.format(result['carType']))
                result_logger.info('---------------------')
                if Debug:
                    saveName = '-'.join([str(result['plateNumber']),
                                         str(result['berthId']), str(result['rfidInfo'])])
                    saveName = os.path.join(saveResult, basename(
                        car_ob.fullName).split('-')[0] + "-" + saveName) + ".jpg"
                    car_roi = car_ob.carBox
                    saveframe_roi = saveframe[car_roi[1]: car_roi[3], car_roi[0]: car_roi[2]]
                    cv2.imwrite(saveName, saveframe_roi, [int(cv2.IMWRITE_JPEG_QUALITY), 60])

            except Exception as err:
                my_logger.error('post result to server occur execption---{}'.format(err))
        else:
            my_logger.info(
                'detect car plate is null , please check {}'.format(car_ob.fullName))
            return
    elif len(history) >= 1 and cur_track_id != pre_track_id:
        if history[-1].plate[1] < 0.95:
            my_logger.info('{} 车牌识别置信度较低'.format(basename(history[-1].fullName)))
            return

        car = history.pop(0)
        history = []

        car_id = car.car_id
        car_ob = car_info[car_id]
        my_logger.info('send_server is {}'.format(car_ob.send_server))
        if car_ob.send_server:
            return

        saveframe = cv2.imread(car_ob.fullName)
        result = post_result(car_ob, saveframe, car_ob.fullName)

        if len(result) > 0:
            try:
                result_carport_queue.put(result)

                car_ob.send_server = True
                car_info[car_id] = car_ob

                my_logger.info(
                    'after send_server: {}'.format(car_ob.send_server))

                result_logger.info('--- detect reuslt ---')

                result_logger.info(
                    'current image name:       {}'.format(car_ob.fullName))
                result_logger.info(
                    'current car plate number: {}'.format(result['plateNumber']))
                result_logger.info(
                    'current berthId number  : {}'.format(result['berthId']))
                result_logger.info(
                    'current rfidInfo:         {}'.format(result['rfidInfo']))
                result_logger.info(
                    'current gps:              {},{}'.format(result['longitude'], result['latitude']
                                                             ))
                result_logger.info(
                    'current car corlor      : {}'.format(result['plateColor']))
                result_logger.info(
                    'current car shape       : {}'.format(result['carType']))
                result_logger.info('---------------------')
                if Debug:
                    saveName = '-'.join([str(result['plateNumber']),
                                         str(result['berthId']), str(result['rfidInfo'])])
                    saveName = os.path.join(saveResult, basename(
                        car_ob.fullName).split('-')[0] + "-" + saveName) + ".jpg"
                    car_roi = car_ob.carBox
                    saveframe_roi = saveframe[car_roi[1]
                                              : car_roi[3], car_roi[0]: car_roi[2]]
                    cv2.imwrite(saveName, saveframe_roi, [
                        int(cv2.IMWRITE_JPEG_QUALITY), 60])

            except Exception as err:
                my_logger.error('post result to server occur execption---{}'.format(err))
        else:
            my_logger.info(
                'detect car plate is null , please check {}'.format(car_ob.fullName))
            return


def post_empty_carport_result(img, rfid_number, gps_info, timeStamp):
    def convert_base64_encode(image):
        """
        input: cvMat
        return: base64encode
        """
        retval, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")

        return "data:image/jpeg;base64," + jpg_as_text

    evt = "evt.scanner.car"
    evtGuid = str(uuid.uuid4())
    deviceId = Device_Id
    my_logger.info('gps info****{}'.format(str(gps_info)))
    longitude = gps_info.split(',')[0][1:]
    latitude = gps_info.split(',')[1][1:]

    dataTime = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(float(timeStamp)))
    try:
        resize_img = cv2.resize(img, (960, 540))

        picFull = convert_base64_encode(resize_img)
        parkingStatus = 0
        json_dict = {"evt": evt, "evtGuid": evtGuid, "deviceId": deviceId, "longitude": longitude, "latitude": latitude,
                     "dataTime": str(dataTime), "rfidInfo": str(rfid_number),
                     "berthId": str(rfid_number), "berthStatus": 0, "berthCredible": 1.0, "platePosition": None,
                     "plateNumber": None, "plateCredible": 0.0, "plateColor": 0, "carType": None, "picPlate": None,
                     "picFull": picFull, "parkingStatus": parkingStatus, "empty": 1}

        return json_dict
    except BaseException as e:
        my_logger.error('prepare result occur exception---{}'.format(e))
        my_logger.error(traceback.format_exc())

        return {}


def post_result(car, img, name=None):
    """
    car: car_class
    """
    global redis_object
    if name:
        my_logger.info('post result to server , image name: {}'.format(name))
    else:
        my_logger.info('post result to server')

    def convert_base64_encode(image):
        """
        input: cvMat
        return: base64encode
        """
        retval, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")

        return "data:image/jpeg;base64," + jpg_as_text

    plate_color = {0: 1, 1: 5, 2: 2, 3: 4, 4: 3}
    car_type = {"small": 0, "big": 1}
    evt = "evt.scanner.car"
    evtGuid = str(uuid.uuid4())
    deviceId = Device_Id

    my_logger.info('car.plate: {}'.format(car.plate))
    try:
        latitude = 107.123456
        if car.plate[0] is not None and car.plate is not None:
            dataTime = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(float(car.timeStamp)))
            longitude = car.gps.split(',')[0][1:]
            latitude = car.gps.split(',')[1][1:]

            rfid = '000000000000'

            image_baseName = basename(name).split('-')[1]
            current_time_stamp, gps_info, rfid_info, _ = image_baseName.split("_")
            current_time_stamp = float(current_time_stamp)
            all_rfid = redis_object.get()

            left, top, right, bottom = car.plate[2]
            center_y = (top + bottom) / 2

            # end_time_stamp = float(current_time_stamp) + 0.5
            for pindex, prfid in enumerate(all_rfid):
                p_time_stamp, gps, rfid_number = prfid.split('_')
                p_time_stamp = float(p_time_stamp)
                my_logger.info('gps: {}'.format(gps))
                if len(gps.split(',')) < 2:
                    gps = 'E40300.00000,N40300.00000'
                longitude = gps.split(',')[0][1:]
                latitude = gps.split(',')[1][1:]
                rfid_number = rfid_number

                if center_y >= 540:
                    if p_time_stamp > current_time_stamp + 0.2 and p_time_stamp < current_time_stamp + 0.5:
                        rfid = rfid_number
                        redis_object.remove_all(value=rfid)
                        break
                    else:
                        rfid = '000000000000'
                else:
                    if p_time_stamp > current_time_stamp + 1 and p_time_stamp < current_time_stamp + 1.3:
                        rfid = rfid_number
                        redis_object.remove_all(value=rfid)
                        break
                    else:
                        rfid = "000000000000"
            if len(all_rfid) == 0:
                rfid = "000000000000"

            plateNumber = car.plate[0]
            plateCredible = car.plate[1]
            plateColor = plate_color[car.plate[3][0]]
            carType = car_type[car.shape[0]]

            verify_plate_etc = "000000000"
            all_plate = redis_object.get(key='plate')
            for pindex, pplate in enumerate(all_plate):
                my_logger.info('pplate: {}'.format(pplate))
                p_time_stamp, plate_etc = pplate.split('_')
                p_time_stamp = float(p_time_stamp)

                if p_time_stamp > current_time_stamp + 0.5 and p_time_stamp < current_time_stamp + 1.5:
                    verify_plate_etc = plate_etc
                    redis_object.remove_all(key='plate', value=plate_etc)
                    break

            if verify_plate_etc[-1] == '-':
                verify_plate_etc = verify_plate_etc[:-1]

            # if verify_plate_etc != '000000000' and verify_plate_etc != plateNumber:
            #     plateNumber = verify_plate_etc
            
            if get_equal_rate_1(verify_plate_etc , plateNumber) > 0.8:
                plateNumber = verify_plate_etc

            car_roi = car.carBox
            cv_img_roi = img[car_roi[1]: car_roi[3], car_roi[0]: car_roi[2]]

            left -= 20
            top -= 20
            right += 100
            bottom += 100

            s_h, s_w = cv_img_roi.shape[:2]

            left = max(left, 0)
            top = max(top, 0)
            right = min(right, s_w)
            bottom = min(bottom, s_h)

            picPlate = convert_base64_encode(
                cv_img_roi[top: bottom, left: right])
            platePosition = {"left": left, "top": top,
                             "right": right, "bottom": bottom}

            if car.ground is not None and car.ground[0] is not None:
                berthId = str(car.ground[0])
                berthCredible = car.ground[1]
                berthStatus = 1
            else:
                berthId = str(0)
                berthCredible = 0
                berthStatus = 2
            if rfid is None:
                berthStatus = 2
                rfid = "000000000000"

            resize_img = cv2.resize(img, (960, 540))
            picFull = convert_base64_encode(resize_img)
            parkingStatus = 0
            json_dict = {"evt": evt, "evtGuid": evtGuid, "deviceId": deviceId, "longitude": longitude,
                         "latitude": latitude, "dataTime": str(dataTime), "rfidInfo": str(rfid),
                         "berthId": str(rfid), "berthStatus": 1, "berthCredible": 1.0, "platePosition": platePosition,
                         "plateNumber": plateNumber, "plateCredible": plateCredible, "plateColor": plateColor,
                         "carType": carType, "picPlate": picPlate,
                         "picFull": picFull, "parkingStatus": parkingStatus, "empty": 0}

            return json_dict
        else:
            return {}
    except BaseException as e:
        my_logger.error('prepare result occur exception---{}'.format(e))
        my_logger.error(traceback.format_exc())
    return {}


def callback(ch, method, properties, body):
    my_logger.info('step into callback')
    path = body.decode("utf-8")
    test_main(path)

    # os.remove(path)


def empty_carport_process(empty_q):
    while True:
        result = empty_q.get()
        empty_q.task_done()
        try:
            my_logger.info("发送空车位-----------------------")
            start_t = time.time()
            my_logger.info("result: {}".format(result))

            # r = requests.post(post_url, headers=headers,
            #                             data=json.dumps(result))errorcode
            r = requests.post(post_url_A, headers=headers,
                              data=json.dumps(result))
            my_logger.info('post empty result cost time: {}'.format(time.time() - start_t))
            if r.status_code == 200:
                my_logger.info("send empty")
                my_logger.info(r.json()['message'])
            else:
                my_logger.info("send empty")
                my_logger.error(r.json()['message'])

        except BaseException as e:
            my_logger.error("send empty carport occur question , switch thread B ")
            r = requests.post(post_url_B, headers=headers,
                              data=json.dumps(result))
            my_logger.info("---------------thread B------------------------")
            if r.status_code == 200:
                my_logger.info("send empty")
                my_logger.info(r.json()['message'])
            else:
                my_logger.info("send empty")
                my_logger.error(r.json()['message'])
            # my_logger.error(traceback.format_exc())


def result_process(result_q):
    while True:
        result = result_q.get()
        result_q.task_done()
        try:
            my_logger.info("发送结果------------------------")
            start_t = time.time()
            r = requests.post(post_url_A, headers=headers, data=json.dumps(result))
            my_logger.info('post server cost time: {}'.format(time.time() - start_t))
            if r.status_code == 200:
                my_logger.info(r.json()['message'])
            else:
                my_logger.error(r.json()['message'])

        except BaseException as e:
            my_logger.error("send empty carport occur question , switch thread B ")
            r = requests.post(post_url_B, headers=headers,
                              data=json.dumps(result))
            my_logger.info("---------------thread B------------------------")
            if r.status_code == 200:
                my_logger.info("send server")
                my_logger.info(r.json()['message'])
            else:
                my_logger.info("send empty")
                my_logger.error(r.json()['message'])


if __name__ == "__main__":
    empty_process = multiprocessing.Process(target=empty_carport_process, args=(empty_carport_queue,))
    result_process = multiprocessing.Process(target=result_process, args=(result_carport_queue,))
    empty_process.start()
    result_process.start()

    while True:
        try:
            channel.basic_consume(
                queue='hello', on_message_callback=callback, auto_ack=True)
            print(' [*] Waiting for messages. To exit press CTRL+C')
            channel.start_consuming()
        except Exception as err:
            my_logger.error('occur exception---{}'.format(err))
            my_logger.error(traceback.format_exc())

            if online_model:
                connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host='vbs-rabbitmq', port=5672, credentials=credentials, heartbeat=0))
            else:
                connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host='localhost', port=5672, heartbeat=0))

            channel = connection.channel()
            channel.queue_declare(queue='hello')
            continue
