import time
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


PIXEL_MEANS = np.array([[[104.0, 117.0, 123.0]]], dtype=np.float32)
BBOX_COLOR = (0, 255, 0)  # green
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('deploy.prototxt', 'vgg.caffemodel', caffe.TEST)
lm_handle = open('labelmap_voc.prototxt', 'r')
lm_map = caffe_pb2.LabelMap()
text_format.Merge(str(lm_handle.read()), lm_map)
cls_dict = {x.label: x.display_name for x in lm_map.item}


def preprocess(src):
    """Preprocess the input image for SSD
    """
    img = cv2.resize(src, (300, 300))
    img = img.astype(np.float32) - PIXEL_MEANS
    return img


def postprocess(img, out):
    """Postprocess the ouput of the SSD object detector
    """
    h, w, c = img.shape
    box = out['detection_out'][0, 0, :,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0, 0, :, 1]
    conf = out['detection_out'][0, 0, :, 2]
    return box.astype(np.int32), conf, cls


def detect(origimg, net):
    img = preprocess(origimg)
    img = img.transpose((2, 0, 1))

    tic = time.time()
    net.blobs['data'].data[...] = img
    out = net.forward()
    dt = time.time() - tic
    box, conf, cls = postprocess(origimg, out)

    return box, conf, cls


def show_bounding_boxes(img, box, conf, cls, cls_dict, conf_th):

    for bb, cf, cl in zip(box, conf, cls):
        cl = int(cl)
        # Only keep non-background bounding boxes with confidence value
        # greater than threshold
        if cl == 0 or cf < conf_th:
            continue
        x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), BBOX_COLOR, 2)
        txt_loc = (max(x_min, 5), max(y_min-3, 20))
        cls_name = cls_dict.get(cl, 'CLASS{}'.format(cl))
        txt = '{} {:.2f}'.format(cls_name, cf)
        cv2.putText(img, txt, txt_loc, cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    BBOX_COLOR, 1)


def process_result(box, conf, cls, cls_dict, conf_th):

    result = []
    char = []
    for bb, cf, cl in zip(box, conf, cls):
        cl = int(cl)

        if cl == 0 or cf < conf_th:
            continue
        x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
        cls_name = cls_dict.get(cl, 'CLASS{}'.format(cl))
        result.append([cls_name, cf, [x_min, y_min, x_max, y_max]])
    res = sorted(result, key=lambda x: x[2][0])
    for index in res:
        char.append(index[0])
    if len(char) == 7:
        stats = 1
        return stats, char
    else:
        stats = 0
        return stats, char


def read_cam_and_detect(img, net, cls_dict, conf_th):

    box, conf, cls = detect(img, net)
    s, r = process_result(box, conf, cls, cls_dict, conf_th)

    return s, r


def main(img):

    caffe.set_device(0)
    caffe.set_mode_gpu()
    box, conf, cls = detect(img, net)
    s, r = process_result(box, conf, cls, cls_dict, 0.7)

    return s, r
