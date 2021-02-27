#coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

import codecs
import sys
import tensorflow as tf
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
# K.set_image_dim_ordering('tf')
K.image_data_format() == 'channels_first'


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
set_session(sess)

import cv2
import numpy as np
import os 
import sys


plateType  = [u"蓝牌",u"黄牌",u"新能源",u"白色",u"黑色"]

current_dirpath  = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dirpath)


def Getmodel_tensorflow(nb_classes):
    # nb_classes = len(charset)

    img_rows, img_cols = 9, 34
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    # x = np.load('x.npy')
    # y = np_utils.to_categorical(range(3062)*45*5*2, nb_classes)
    # weight = ((type_class - np.arange(type_class)) / type_class + 1) ** 3
    # weight = dict(zip(range(3063), weight / weight.mean()))  # 调整权重，高频字优先

    model = Sequential()
    model.add(Conv2D(16, (5, 5),input_shape=(img_rows, img_cols,3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model_color = Getmodel_tensorflow(5)
g3 = tf.get_default_graph()
with g3.as_default():
    model_color.load_weights(os.path.join(current_dirpath , "model/plate_type.h5"))
# model.save("model/plate_type.h5")

def SimplePredict(image):
    image = cv2.resize(image, (34, 9))
    image = image.astype(np.float) / 255
    with g3.as_default():
        set_session(sess)
        res = np.array(model_color.predict(np.expand_dims(image , axis=0))[0])
    return plateType[res.argmax()] , res[res.argmax()]

if __name__ == "__main__":
    img = cv2.imread("WechatIMG205.jpeg")
    res = SimplePredict(img)
    print(res)
    


