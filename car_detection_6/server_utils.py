import torch
import cv2
import numpy as np
from utils.utils import non_max_suppression


def preprocess(image, target_size=640, gt_boxes=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    ih, iw = target_size, target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    image_paded = np.transpose(image_paded[np.newaxis, ...], (0, 3, 1, 2))
    if gt_boxes is None:
        return image_paded
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def infer(img, model, device, thres=0.5):
    # Get names and colors
    # Run inference
    img = img.float()
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    pred = model(img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, thres, 0.5, fast=True)
    return pred


def postprocess(pred, org_img_shape, input_size=640):
    bboxes = []
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2
    for tensor in pred:
        tensor = tensor.detach().cpu().numpy()
        tensor[:, 0] = 1.0 * (tensor[:, 0] - dw) / resize_ratio
        tensor[:, 2] = 1.0 * (tensor[:, 2] - dw) / resize_ratio
        tensor[:, 1] = 1.0 * (tensor[:, 1] - dh) / resize_ratio
        tensor[:, 3] = 1.0 * (tensor[:, 3] - dh) / resize_ratio
        for item in tensor:
            bboxes.append(item)

    return bboxes

