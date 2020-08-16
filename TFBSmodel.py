#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time      :2020/5/12 11:45
# @Author    :Lingbo Yang
# @Version   :v1.0


import os,shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
import random
from keras import backend as K
import gdal,gdalconst
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import datetime
from keras.callbacks.tensorboard_v2 import TensorBoard
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, cross_val_score



matplotlib.use("Agg")
K.image_data_format()
K.set_image_data_format('channels_first')

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

img_w = 128
img_h = 128
img_c = 18
# 有一个为背景
n_label = 1
# directory where the src and label folders is
filepath = 'E:/TFBS/Dataset/'
EPOCHS = 30
BS = 16


# +++++++++++++++++++++++++++++++++++++++++++
#     custom metrics v1.0
# +++++++++++++++++++++++++++++++++++++++++++
def recall(y_true, y_pred): # producer accuracy
     TP = K.sum(y_true * K.round(y_pred))
     recall = TP / (K.sum(y_true) + K.epsilon()) # equivalent to the above two lines of code
     return recall

def precision(y_true, y_pred): #user accuracy
    TP = K.sum(y_true * K.round(y_pred))
    precision = TP / (K.sum(K.round(y_pred))+ K.epsilon()) # equivalent to the above two lines of code
    return precision


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    TP = K.sum(y_true * K.round(y_pred))
    precision = TP / (K.sum(K.round(y_pred))+ K.epsilon())
    recall = TP / (K.sum(K.round(y_true))+ K.epsilon())
    F1score = 2 * precision * recall / (precision + recall+K.epsilon())
    return F1score

def kappa_metrics(y_true, y_pred):
    # Calculates the kappa coefficient
    TP = K.sum(y_true * K.round(y_pred))
    FP = K.sum((1 - y_true) * K.round(y_pred))
    FN = K.sum(y_true * (1 - K.round(y_pred)))
    TN = K.sum((1 - y_true) * (1 - K.round(y_pred)))
    totalnum=TP+FP+FN+TN
    p0 = (TP + TN)/ totalnum
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN))/ (totalnum * totalnum)
    kappa_coef = (p0 - pe)/(1 - pe + K.epsilon())
    return kappa_coef


def OA(y_true, y_pred):  # producer accuracy
    TP = K.sum(y_true * K.round(y_pred))
    FP = K.sum((1 - y_true) * K.round(y_pred))
    FN = K.sum(y_true * (1 - K.round(y_pred)))
    TN = K.sum((1 - y_true) * (1 - K.round(y_pred)))
    totalnum = TP + FP + FN + TN
    overallAC=(TP+TN)/totalnum
    return overallAC

# +++++++++++++++++++++++++++++++++++++++++++
#     load image based on gdal v1.0
# +++++++++++++++++++++++++++++++++++++++++++
def load_img(path, grayscale=False,scaled=True,GrayScaled=True):
    inputdst = gdal.Open(path)
    if inputdst is None:
        print("can't open " + path)
        return False

    # get the metadata of data
    X_width = inputdst.RasterXSize
    X_height = inputdst.RasterYSize


    if grayscale:  # if only one band in image
        src_img = inputdst.ReadAsArray(
            xoff=0, yoff=0, xsize=X_width, ysize=X_height)
        if GrayScaled:
            src_img = src_img.reshape(1, X_height, X_width)
    else:
        src_img = inputdst.ReadAsArray(
            xoff=0, yoff=0, xsize=X_width, ysize=X_height)
        if scaled==True:
            src_img = (-np.array(src_img, dtype="float32")) / 4000.
    return src_img


# get train and validation dataset
def get_train_val(val_rate=0.25):
    """
    v1
    :param val_rate:
    :return:
    """
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(filepath + 'src'):
        filePath_tmp = os.path.join(filepath + 'src', pic)
        if os.path.isdir(filePath_tmp):
            continue
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set




def get_image_set(path):
    train_url = []
    for pic in os.listdir(path + 'src'):
        filePath_tmp = os.path.join(path + 'src', pic)
        if os.path.isdir(filePath_tmp):
            continue
        train_url.append(pic)
    random.shuffle(train_url)
    return train_url

def generateData(batch_size, data=[]):
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'src/' + url)
            train_data.append(img)
            label = load_img(filepath + 'label/' + url, grayscale=True)
            train_label.append(label)
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0



def generateValidData(batch_size, data=[]):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'src/' + url)
            valid_data.append(img)
            label = load_img(filepath + 'label/' + url, grayscale=True)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0



def TFBS():
    # [bs, 18, 128, 128]
    inputs = keras.layers.Input((18, img_h,img_w))

    def reshapes(embed):
        # => [BS, 2, 9, 128, 128]
        embed = tf.reshape(embed, [BS, 2, 9, img_h, img_w])
        # => [BS, 128, 128, 9, 2]
        embed = tf.transpose(embed, [0, 3, 4, 2, 1])
        # => [BS*128*128, 9, 2]
        embed = tf.reshape(embed, [BS*128*128, 9, 2])
        return embed
    # => [BS*128*128, 9, 2]
    inputs1 = keras.layers.Lambda(reshapes)(inputs)

    # => [bs * 128 * 128, 64]
    lstm1 = keras.layers.LSTM(64)(inputs1)
    def reshapes1(embed):
    # => [BS, 128,128, 64]
        embed = tf.reshape(embed, [BS,128,128, 64])
        # => [BS, 64, 128, 128]
        embed = tf.transpose(embed, [0, 3, 1, 2])
        return embed
    # => [bs, 64, 128, 128] BS1指的是原来的BS除以128*128之后的BS,此处为1
    inputs2 = keras.layers.Lambda(reshapes1)(lstm1)

    # [bs, 64, 256, 256]=>[bs, 32, 256, 256]
    conv1 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs2)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    # [bs, 32, 256, 256]=>[bs, 32, 128, 128]
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # [bs, 32, 128, 128]=>[bs, 64, 128, 128]
    conv2 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    # [bs, 64, 128, 128]=>[bs, 64, 64, 64]
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # [bs, 64, 128, 128]=>[bs, 128, 64, 64]
    conv3 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    # [bs, 128, 64, 64]=>[bs, 128, 32, 32]
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # [bs, 128, 32, 32]=>[bs, 256, 32, 32]
    conv4 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    # [bs, 256, 32, 32]=>[bs, 256, 16, 16]
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # [bs, 256, 16, 16]=>[bs, 512, 16, 16]
    conv5 = keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

    # [bs, 512, 32, 32] + [bs, 256, 32, 32]=>[bs, 768, 32, 32]
    up6 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    # [bs, 768, 32, 32] => [bs, 256, 32, 32]
    conv6 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    # [bs, 256, 32, 32] => [bs, 256, 64, 64] + [bs, 128, 64, 64]=>[bs, 384, 64, 64]
    up7 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    # [bs, 384, 64, 64] = > [bs, 128, 64, 64]
    conv7 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    # [bs, 128, 64, 64] = > [bs, 128, 128, 128] + [bs, 64, 128, 128] =>[bs, 192, 128, 128]
    up8 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    # [bs, 192, 128, 128] => [bs, 64, 128, 128]
    conv8 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    # [bs, 64, 128, 128] => [bs, 64, 256, 256] + [bs, 32, 256, 256] => [bs, 96, 256, 256]
    up9 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    # [bs, 96, 256, 256] => [bs, 32, 256, 256]
    conv9 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    # [bs, 32, 256, 256] => [bs, 1, 256, 256]
    conv10 = keras.layers.Conv2D(n_label, (1, 1), activation="sigmoid")(conv9)

    model = keras.Model(inputs=inputs, outputs=conv10)
    model.compile(
        optimizer='Adam',
        loss='binary_crossentropy',
        metrics=['accuracy',recall,precision,kappa_metrics,fmeasure,OA]
        )


    return model


def CrossValidation():

    train_set, _ = get_train_val(val_rate=0.0)
    seed = 41
    np.random.seed(seed)
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_nb = 0
    for train_index, test_index in kf.split(train_set):
        train_set = np.array(train_set)
        train_split_set, test_split_set = train_set[train_index], train_set[test_index]
        train_split_set = train_split_set.tolist()
        test_split_set = test_split_set.tolist()

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = filepath + 'model//tfbs_AR2019_CV' + str(cv_nb) + '_' + current_time +'_weights.h5'
        log_dir = 'logs\\tfbs_AR2019_EP' + str(EPOCHS) + '_cv' + str(cv_nb) + '_' + current_time
        tensorboard = TensorBoard(log_dir=log_dir)

        model = TFBS()
        modelcheck = ModelCheckpoint(
            model_name,
            monitor='val_loss',
            save_best_only=True
            )
        callable = [modelcheck,tensorboard]

        train_numb = len(train_split_set)
        valid_numb = len(test_split_set)
        print("the number of train data is", train_numb)
        print("the number of val data is", valid_numb)
        model.summary()
        with tf.device('/gpu:0'):
            H = model.fit_generator(
                generator=generateData(
                    BS,
                    train_split_set),
                steps_per_epoch=train_numb // BS,
                epochs=EPOCHS,
                verbose=1,
                validation_data=generateValidData(
                    BS,
                    test_split_set),
                validation_steps=valid_numb // BS,
                callbacks=callable,
                max_queue_size=1)
        cv_nb = cv_nb+1


if __name__ == '__main__':

    # directory where the src and label folders is
    filepath = 'E:/TFBS/Dataset/'
    EPOCHS = 30
    BS = 16

    # 10-fold cross validation
    CrossValidation()

