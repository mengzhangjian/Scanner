# from .hyperlpr_py3 import e2e

# from hyperlpr_py3 import e2e
# from keras.backend.tensorflow_backend import set_session
# import tensorflow as tf
from flask import Flask, Blueprint
from flask import request
import json
import cv2
import numpy as np
import os
from hyperlpr import *
import codecs
import sys
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

api_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../api")
sys.path.append(api_path)
from log import get_logger

logger = get_logger('log/webplate.log')

# graph = tf.Graph()
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
# sess = tf.Session(config=config , graph=graph)
# sess.run(tf.global_variables_initializer())
# graph = tf.get_default_graph()

# set_session(sess)

car_plate_server = Blueprint('plate_server', __name__)

def car_plate_detect(cv_img, name=None):
    if name:
        logger.info('{} :begin car plate detect , process'.format(name))
    else:
        logger.info('begin car plate detect , process')

    global graph
    global sess
    res = {'results': []}
    val_image = cv_img

    # with graph.as_default():
    #     set_session(sess)
      # result = e2e.SimpleRecognizePlateByE2E(val_image)
    result = HyperLPR_plate_recognition(val_image)
    if len(result) > 0:
        result = result[0]
    else:
        return res

    logger.info('car detect result: {}'.format(result))
    if len(result) > 0 and result[1] > 0.5 and len(result[0]) >= 7:
            # print(result[0], result[1])
            #print(result[1], "Hello")
            # left, top, w, h = result[2]
            # right = left + w
            # bottom = top + h

        left, top, right , bottom = result[2]

        res['results'].append({
                'name': str(result[0]),
                'score': float(result[1]),
                "location":
                    {
                        "left": int(left),
                        "top": int(top),
                        "right": int(right),
                        "bottom": int(bottom)
                }
            })

    logger.info('car detect final result: {}'.format(res))
    logger.info('-----------------------------------------')
    return res


# if __name__ == '__main__':
#    app.run(host='0.0.0.0', port=8083, debug=False)

if __name__ == "__main__":
    # 读入图片
    image = cv2.imread(
        "20200723103200scannerImage_3Pnt.jpg")
    # 识别结果
    res = HyperLPR_plate_recognition(image)
    print(res)

    for r in res:
        plate_code, conf, box = r[0] , r[1] , r[2]

        cv2.rectangle(image, (box[0], box[1]) , (box[2] , box[3]) , (0 , 255 , 0) , 2)

    cv2.imwrite("save.jpg", image)
