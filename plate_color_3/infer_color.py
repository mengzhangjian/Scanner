import os
os.environ['GLOG_minloglevel'] = '2'
import sys
sys.path.append("/root/caffe-ssd/python")
import caffe
import numpy as np
import cv2
abs_path = os.path.dirname(__file__)
labels = np.loadtxt(os.path.join(abs_path, 'model_color/synset.txt'), str, delimiter='\t')
gpu_id = 0


def initial():
    """
    :return:
    """

    deploy_path = os.path.join(abs_path, 'model_color/test.prototxt')
    model_path = os.path.join(abs_path, 'model_color/model.caffemodel')
    mean_path = os.path.join(abs_path, 'model_color/train_mean.npy')
    caffe.set_device(int(gpu_id))
    caffe.set_mode_gpu()
    net = caffe.Net(str(deploy_path), str(model_path), caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    mu = np.load(mean_path)
    mu = mu.mean(1).mean(1)
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    net.blobs['data'].reshape(1, 3, 64, 64)
    return net, transformer


net, transformer = initial()


def extractFeature(image, net, transformer):
    """
    :param image:
    :param net:
    :param transformer:
    :return:
    """
    image = image / 255.
    image = image[:, :, (2, 1, 0)]
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    out = net.forward()
    output_prob = out['prob'][0]

    return output_prob


def test_topk(im):
    """
    :param im:
    :return:
    """
    out = extractFeature(im, net, transformer)
    predict = labels[out.argmax()]
    prob = out[out.argmax()][0][0]
    return prob, predict


def get_result(cv_img):
    """
    :param cv_img:
    :return:
    """
    caffe.set_device(0)
    caffe.set_mode_gpu()
    score, name = test_topk(cv_img)

    return score, name

#if __name__ == "__main__":
#    im = cv2.imread("1.jpg")
#    print(test_topk(im))



