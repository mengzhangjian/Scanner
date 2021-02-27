import requests
import base64
import json
import cv2
import numpy as np
import math
import time
import os
path = "./save"


def recognizeOne(src):
    x_tempx = src
    x_temp = cv2.resize(x_tempx, (160, 40))
    x_temp = x_temp.transpose(1, 0, 2)
    y_pred = pred_model.predict(np.array([x_temp]))
    y_pred = y_pred[:, 2:, :]

    return fastdecode(y_pred)


def rotate_about_center2(src, radian, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    angle = radian * 180 / np.pi
    # now calculate new image width and height
    nw = (abs(np.sin(radian) * h) + abs(np.cos(radian) * w)) * scale
    nh = (abs(np.cos(radian) * h) + abs(np.sin(radian) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_CUBIC)


def get_group(arr):
    radian_45 = np.pi / 4
    radian_90 = np.pi / 2
    radian_135 = radian_45 * 3
    radian_180 = np.pi
    ret_arr = [[], [], [], []]
    for i in range(len(arr)):
        if arr[i] < radian_45:
            ret_arr[0].append(arr[i])
        elif arr[i] < radian_90:
            ret_arr[1].append(arr[i])
        elif arr[i] < radian_135:
            ret_arr[2].append(arr[i])
        else:
            ret_arr[3].append(arr[i])

    while [] in ret_arr:
        ret_arr.remove([])

    # print ret_arr
    return ret_arr


def get_min_var_avg(arr):
    group_arr = get_group(arr)
    var_arr = []

    var_arr = [np.var(group_arr[i]) for i in range(len(group_arr))]
    min_var = 10000
    min_i = 0
    for i in range(len(var_arr)):
        if var_arr[i] < min_var:
            min_var = var_arr[i]
            min_i = i
    # print min_var, i
    avg = np.mean(group_arr[min_i])
    return avg


def get_rotate_radian(radian, reverse=False):
    radian_45 = np.pi / 4
    radian_90 = np.pi / 2
    radian_135 = radian_45 * 3
    radian_180 = np.pi
    ret_radian = 0
    if radian < radian_45:
        ret_radian = radian
    elif radian < radian_90:
        ret_radian = radian - radian_90
    elif radian < radian_135:
        ret_radian = radian - radian_90
    else:
        ret_radian = radian - radian_180

    if reverse:
        ret_radian += radian_90
    return ret_radian


MAX_WIDTH = 1000
Min_Area = 2000


def rotate(img):
    """"""
    picHeight, picWidth = img.shape[: 2]
    if picHeight > MAX_WIDTH:
        img = cv2.resize(img, (int(picWidth / 2), int(picHeight / 2)), interpolation=cv2.INTER_AREA)
    oldimg = img
    # cvtColor to Ycrcb and equlize channel y
    ycb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycb)
    equ = cv2.equalizeHist(y)
    merge = cv2.merge((equ, cr, cb))

    merult = cv2.cvtColor(merge, cv2.COLOR_YCrCb2BGR)
    blur = 3
    if blur > 0:
        img = cv2.GaussianBlur(merult, (blur, blur), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dilate = cv2.dilate(img, element)
    erode = cv2.erode(img, element)

    result = cv2.absdiff(dilate, erode)

    retval, result = cv2.threshold(result, 60, 255, cv2.THRESH_BINARY)  # 40 for plate  100 for boweihao

    result = cv2.bitwise_not(result)
    bw = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15,
                               -2)  # 15 -2 for car plate
    horizontal = bw.copy()
    hoHieght, hoWidth = horizontal.shape[:2]

    horizontalSize = hoWidth / 30

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizontalSize), 1))  # for car plate
    hoErode = cv2.erode(horizontal, horizontalStructure)
    hoDilate = cv2.dilate(hoErode, horizontalStructure)

    vertical = bw.copy()
    verHieght, verWidth = vertical.shape[:2]
    verticalSize = verHieght / 30
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(verticalSize)))
    verErode = cv2.erode(vertical, verticalStructure)
    verDilate = cv2.dilate(verErode, verticalStructure)

    lines = cv2.HoughLines(hoDilate, 1, np.pi / 180, 200)  # for car plate detect lines

    if lines is not None:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # cv2.line(oldimg,(x1,y1),(x2,y2),(0,0,255),2)
        l = len(lines[0])
        theta_arr = [lines[0][i][1] for i in range(l)]
        rotate_theta = get_min_var_avg(theta_arr)
        img2 = rotate_about_center2(oldimg,
                                    get_rotate_radian(rotate_theta, oldimg.shape[0] > oldimg.shape[1]))  # hight > width
        return img2

    return img

import  uuid

for index in os.listdir(path):

    name = os.path.join(path, index) 
    print(name)
    img = cv2.imread(name)
    #img = rotate(img)
    #cv2.imwrite(str(uuid.uuid4()) + ".jpg", img)
    imencoded = cv2.imencode('.jpg', img)[1].tostring()
    r = requests.post('http://0.0.0.0:8084/detect', data=imencoded, timeout=50)
    rs = json.loads(r.text)
    stats = rs["results"]["state"]
    if r.status_code == 200 and stats == 1:

        ground_code = rs["results"]["code"]
        score = rs["results"]["score"]
        print(ground_code)
        print(score)
