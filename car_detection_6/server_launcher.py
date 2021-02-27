#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2021/1/25
"""
import torch
import numpy as np
from flask import Flask, request, Blueprint
import json
import cv2
import os
from server_utils import preprocess, infer, postprocess

from utils.utils import non_max_suppression

abs_path = os.path.dirname(__file__)


weights = os.path.join(abs_path, 'weight', 'best_weight.pt')

device = torch.device('cuda:0')

model = torch.load(weights, map_location="cpu")[
    'model'].float()  # load to FP32
model.fuse()
model.to(device)
model.eval()

name = ['car', 'truck']


def detection_func(img, threshold=0.4):
    result = {"results": []}
    try:
        with torch.no_grad():
            h, w = img.shape[:2]
            val_image = torch.from_numpy(preprocess(img, 640)).to(device)
            pred = infer(val_image, model, device, threshold)
            bboxes = postprocess(pred, (h, w), 640)

            for item in bboxes:
                left, top, right, bottom, score, classes = item
                result['results'].append({
                    'location': {
                        'left': int(left),
                        'top': int(top),
                        'right': int(right),
                        'bottom': int(bottom),
                        'area': (bottom - top) * (right - left)
                    },
                    'name': name[int(classes)],
                    'score': float(score),
                })
        print(result)
        return result
    except Exception as err:
        print(err)
        return {"results": []}


detection_server = Blueprint("detection_server", __name__)
@detection_server.route('/detect', methods=['GET', 'POST'])
def detection():
    try:
        val_image = cv2.imdecode(np.frombuffer(
            request.data, np.uint8), cv2.IMREAD_COLOR)
        result = detection_server(val_image)
        return json.dumps(result), 200
    except Exception as err:
        print(err)
        return json.dumps({"results": []}), 403


def draw_result(cv_img, det_res):
    results = det_res['results']
    for det in results:
        location = det['location']
        conf = det['score']
        cls = det['name']

        xmin = int(location['left'])
        ymin = int(location['top'])
        xmax = int(location['right'])
        ymax = int(location['bottom'])

        cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        s_anno = "{} : {}".format(cls, conf)
        cv2.putText(cv_img, s_anno, (xmin, ymin + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return cv_img


if __name__ == '__main__':
    img_path = "/work/yolov5-pruning-distillation/dataset/project/images/1510.jpg"
    original_image = cv2.imread(img_path)
    detect_res = detection_func(original_image)
    print(detect_res)

    draw_img = draw_result(original_image, detect_res)
    cv2.imwrite("render_img/save.jpg", draw_img)
