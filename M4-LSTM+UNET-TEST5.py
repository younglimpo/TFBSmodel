#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time      :2020/5/12 11:45
# @Author    :Lingbo Yang
# @Version   :v1.0
# @License: BSD 3 clause
# @Description: original is based on Keras, now I changed it to tf.keras to solve the problem of tensorboard
# v3 会出现由于样本量太少，不洗牌直接训练无法有效收敛的问题
# v4 通过对包含正样本的影像进行扩增，从而提高模型训练能力，防止模型衰退
# V5 不进行模型扩增，只在UNET前加一层LSTM64，即可得到很好的效果


import os,shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
import numpy as np
import keras
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input,concatenate
#from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint
#from tensorflow.keras.models import Model
#import matplotlib.pyplot as plt
import random
#from tensorflow.keras import backend as K
from keras import backend as K
import gdal,gdalconst
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import datetime
from keras.callbacks.tensorboard_v2 import TensorBoard
import tensorflow as tf
#import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split, KFold, cross_val_score
# if not os.path.exists(log_dir):
#    os.makedirs(log_dir)


matplotlib.use("Agg")
#import matplotlib.pyplot as plt
K.image_data_format()
K.set_image_data_format('channels_first')

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#seed = 7
#np.random.seed(seed)

img_w = 128
img_h = 128
img_c = 18
# 有一个为背景
n_label = 1
# directory where the src and label folders is
filepath = 'E:/11-ArMsMoTn/04-Dataset/01-train/'
Augmenatation_path='augmentation/' # 增广后训练数据目录，在src文件夹下
validationpath = 'E:/11-ArMsMoTn/04-Dataset/01-train/'#'E:/07-keras/04-UNET-LSTM2D/02-clipraw/02-test/'
EPOCHS = 30
BS = 16


# +++++++++++++++++++++++++++++++++++++++++++
#     custom metrics v1.0
# +++++++++++++++++++++++++++++++++++++++++++
def recall(y_true, y_pred): # producer accuracy
     TP = K.sum(y_true * K.round(y_pred))
     #FN = K.sum(y_true * (1 - K.round(y_pred)))
     # recall = TP / (TP + FN)
     recall = TP / (K.sum(y_true) + K.epsilon()) # equivalent to the above two lines of code
     return recall

def precision(y_true, y_pred): #user accuracy
    TP = K.sum(y_true * K.round(y_pred))
    #FP = K.sum((1 - y_true) * K.round(y_pred))
    #precision = TP / (TP + FP)
    precision = TP / (K.sum(K.round(y_pred))+ K.epsilon()) # equivalent to the above two lines of code
    return precision


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    TP = K.sum(y_true * K.round(y_pred))
    # FP = K.sum((1 - y_true) * K.round(y_pred))
    # FN = K.sum(y_true * (1 - K.round(y_pred)))
    # TN = K.sum((1 - y_true) * (1 - K.round(y_pred)))
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
    X_width = inputdst.RasterXSize  # 原始栅格矩阵的列数
    X_height = inputdst.RasterYSize  # 原始栅格矩阵的行数
    X_bands = inputdst.RasterCount  # 原始波段数
    X_eDT = inputdst.GetRasterBand(1).DataType  # 数据的类型

    if grayscale:  # if only one band in image
        src_img = inputdst.ReadAsArray(
            xoff=0, yoff=0, xsize=X_width, ysize=X_height)
        if GrayScaled:
            src_img = src_img.reshape(1, X_height, X_width)
    else:
        src_img = inputdst.ReadAsArray(
            xoff=0, yoff=0, xsize=X_width, ysize=X_height) #[c, h, w] c = 18对于研究区
        if scaled==True:
            src_img = (-np.array(src_img, dtype="float32")) / 4000.
    return src_img


#filepath = './unet_train/'

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
        filePath_tmp = os.path.join(filepath + 'src', pic)  # 排除文件夹
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




#获取路径下所有的文件
def get_image_set(path):
    train_url = []
    for pic in os.listdir(path + 'src'):
        filePath_tmp = os.path.join(path + 'src', pic)  # 排除文件夹
        if os.path.isdir(filePath_tmp):
            continue
        train_url.append(pic)
    random.shuffle(train_url)
    return train_url

# data for training


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
            # img = img_to_array(img) #from [row, column, channel] to [channel,
            # row, column]
            train_data.append(img)
            label = load_img(filepath + 'label/' + url, grayscale=True)
            #label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0

            # data for validation


def generateValidData(batch_size, data=[],path=filepath):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'src/' + url)
            #img = img_to_array(img)
            valid_data.append(img)
            label = load_img(filepath + 'label/' + url, grayscale=True)
            #label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0

def rotate_90(xb, yb):
    xb = np.transpose(xb,axes=(0,2,1)) #np.transpose(xb,axes=(0,2,1))
    xb = np.flip(xb,2)

    yb = np.transpose(yb)
    yb = np.flip(yb,1)
    return xb, yb

def rotate_180(xb, yb):
    xb = np.transpose(xb,axes=(0,2,1)) #np.transpose(xb,axes=(0,2,1))
    xb = np.flip(xb,2)
    xb = np.transpose(xb, axes=(0, 2, 1))  # np.transpose(xb,axes=(0,2,1))
    xb = np.flip(xb, 2)

    yb = np.transpose(yb)
    yb = np.flip(yb,1)
    yb = np.transpose(yb)
    yb = np.flip(yb, 1)
    return xb, yb


def rotate_270(xb, yb):
    xb = np.transpose(xb,axes=(0,2,1)) #np.transpose(xb,axes=(0,2,1))
    xb = np.flip(xb,2)
    xb = np.transpose(xb, axes=(0, 2, 1))  # np.transpose(xb,axes=(0,2,1))
    xb = np.flip(xb, 2)
    xb = np.transpose(xb, axes=(0, 2, 1))  # np.transpose(xb,axes=(0,2,1))
    xb = np.flip(xb, 2)

    yb = np.transpose(yb)
    yb = np.flip(yb,1)
    yb = np.transpose(yb)
    yb = np.flip(yb, 1)
    yb = np.transpose(yb)
    yb = np.flip(yb, 1)
    return xb, yb

def add_noise3(img):
    for i in range(int(img.shape[1]*img.shape[2]/50)):  # 添加点噪声
        for ii in range(img.shape[0]):
            temp_x = np.random.randint(0, img.shape[1])
            temp_y = np.random.randint(0, img.shape[2])
            img[ii,temp_x,temp_y] = np.random.randint(-3000, -1500)
    return img

def saveArray2disk(x,y,filename):
    """
    #将np数组保存到文件夹中
    :param x: train_x np数组 [c,h,w]
    :param y: 标签 y np数组 [h,w]
    :param filename:文件名 str，如 '1','2'
    :return: 文件名
    """
    tifDriver = gdal.GetDriverByName("GTiff");  # 目标格式
    rstDst = tifDriver.Create((filepath+'src/'+Augmenatation_path+filename+'.tiff'), img_w, img_h, img_c, gdalconst.GDT_Int16);  # 创建目标文件
    for ii in range(img_c):
        rstDst.GetRasterBand(ii + 1).WriteArray(x[ii])  # 写入数据
    rstlabelDst = tifDriver.Create((filepath+'label/'+Augmenatation_path+filename+'.tiff'), img_w, img_h, 1,
                                   gdalconst.GDT_Byte);  # 创建目标文件
    rstlabelDst.GetRasterBand(1).WriteArray(y)  # 写入label数据
    return (Augmenatation_path+filename+'.tiff')

# positive sample augmentation
def sample_augment(data=[],threshold=1000):
    """
    扩增正样本标签
    :param data: 数据列表list ['1',[2]...]
    :param threshold: 正样本像素点数量超过多少时会扩增
    :return: 扩增后的包含原样本的扩增样本
    """
    # print 'generateData...'
    Augmentated_data = []
    # 删除原来的临时增广文件，创建新的增广文件夹
    if os.path.exists(filepath+'src/'+Augmenatation_path):
        shutil.rmtree(filepath+'src/'+Augmenatation_path)
        shutil.rmtree(filepath + 'label/' + Augmenatation_path)
    os.mkdir(filepath + 'src/' + Augmenatation_path)
    os.mkdir(filepath + 'label/' + Augmenatation_path)

    g_count=0 #增广文件名，即1,2,3,4，等
    for i in (range(len(data))):
        print(i)
        url = data[i]
        Augmentated_data.append(url)
        # [band, row, column]
        train = load_img(filepath + 'src/' + url, scaled=False)
        # [row, column]
        label = load_img(filepath + 'label/' + url, grayscale=True,GrayScaled=False)

        if np.sum(label) < threshold: #当图片中正样本数量超过400个时扩增
            continue
        else:
            rdnum = np.random.randint(0, 3) #随机生成0-4的整数
            # 旋转90度
            if rdnum == 0:
                xb, yb = rotate_90(train, label)
                #xb = add_noise3(xb)  # add noise

            # 旋转180度
            if rdnum == 1:
                xb, yb = rotate_180(train, label)
                #xb = add_noise3(xb)  # add noise

            # 旋转270度
            if rdnum == 2:
                xb, yb = rotate_270(train, label)
                #xb = add_noise3(xb)  # add noise

            rdscr2 = np.random.randint(-100, 101) / 1000.0 + 1.0  # 随机生成0-4的整数
            xb = np.int16(xb * rdscr2)
            Augmentated_data.append(saveArray2disk(xb, yb, str(g_count)))
            g_count = g_count + 1

            rdnum = np.random.randint(0, 2)  # 随机生成0-4的整数
            if rdnum == 0:
                # 沿x轴翻转对称
                xb = np.flip(train, 1)  # flipcode = 1：沿x轴翻转
                yb = np.flip(label, 0)
                #xb = add_noise3(xb)  # add noise

            if rdnum == 1:
                # 沿y轴翻转对称
                xb = np.flip(train, 2)  # flipcode = 1：沿x轴翻转
                yb = np.flip(label, 1)
                #xb = add_noise3(xb)  # add noise

            rdscr2 = np.random.randint(-100, 101) / 1000.0 + 1.0  # 随机生成0-4的整数
            xb = np.int16(xb * rdscr2)
            Augmentated_data.append(saveArray2disk(xb, yb, str(g_count)))
            g_count = g_count + 1

            print('已扩增%d' % g_count)
    return Augmentated_data


def LSTM_unet1():
    # [bs, 18, 128, 128]
    inputs = keras.layers.Input((18, img_h,img_w))  # 输入9个波段的数据

    def reshapes(embed):
        # => [BS, 2, 9, 128, 128]
        embed = tf.reshape(embed, [BS, 2, 9, img_h, img_w])
        # => [BS, 128, 128, 9, 2]
        embed = tf.transpose(embed, [0, 3, 4, 2, 1])
        # => [BS*128*128, 9, 2]
        embed = tf.reshape(embed, [BS*128*128, 9, 2])
        return embed
    # => [BS*128*128, 9, 2] BS1指的是原来的BS除以128*128之后的BS,此处为1
    inputs1 = keras.layers.Lambda(reshapes)(inputs)

    # => [bs * 128 * 128, 64]
    #lstm1 = keras.layers.LSTM(64, return_sequences=True)(inputs1)
    lstm2 = keras.layers.LSTM(64)(inputs1)
    def reshapes1(embed):
    # => [BS, 128,128, 64]
        embed = tf.reshape(embed, [BS,128,128, 64])
        # => [BS, 64, 128, 128]
        embed = tf.transpose(embed, [0, 3, 1, 2])
        return embed
    # => [bs, 64, 128, 128] BS1指的是原来的BS除以128*128之后的BS,此处为1
    inputs2 = keras.layers.Lambda(reshapes1)(lstm2)

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
    #

    model = keras.Model(inputs=inputs, outputs=conv10)
    model.compile(
        optimizer='Adam',
        loss='binary_crossentropy',
        metrics=['accuracy',recall,precision,kappa_metrics,fmeasure,OA]
        )


    return model



def train2(Augment = True):

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

        # 样本增广 只针对包含正样本的影像 Augmentation for positive samples
        if Augment == True:
            train_split_set = sample_augment(train_split_set)
            random.shuffle(train_split_set)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = filepath + '02-model//LSTMUNET//LSTMUNET_rice_AR2019_30_CV' + str(cv_nb) + '_' + current_time +'_weights.h5'
        log_dir = 'logs\\M4_LSTM64-UNET_AR2019_EP' + str(EPOCHS) + '_cv' + str(cv_nb) + '_' + current_time
        tensorboard = TensorBoard(log_dir=log_dir)

        model = LSTM_unet1()
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
        with tf.device('/gpu:1'):
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
                callbacks=callable,#class_weight=[0.01,0.99]
                max_queue_size=1)
        cv_nb = cv_nb+1


def train_totalyear(test_path,Augment = False):
    train_set, _ = get_train_val(val_rate=0.0)
    test_set = get_image_set(test_path)

    # 样本增广 只针对包含正样本的影像 Augmentation for positive samples
    if Augment == True:
        train_set = sample_augment(train_set,threshold=100)
        random.shuffle(train_set)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = 'E://11-ArMsMoTn//05-model//02-model//01-finalmodel//LSTMUNET_arg_TrainAR19_TESTAR18_EP30_' + current_time + '_weights.h5'
    log_dir = 'E:\\11-ArMsMoTn\\05-model\\02-model\\01-finalmodel\\logs\\M4_LSTMUNET_AUG_TrainAR19_testAR18_EP' + str(
        EPOCHS) + '_' + current_time

    tensorboard = TensorBoard(log_dir=log_dir)



    model = LSTM_unet1()
    modelcheck = ModelCheckpoint(
        model_name,
        monitor='val_loss',
        save_best_only=True
        )
    callable = [modelcheck,tensorboard]

    train_numb = len(train_set)
    valid_numb = len(test_set)
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)
    model.summary()
    with tf.device('/gpu:1'):
        H = model.fit_generator(
            generator=generateData(
                BS,
                train_set),
            steps_per_epoch=train_numb // BS,
            epochs=EPOCHS,
            verbose=1,
            validation_data=generateValidData(
                BS,
                test_set, path=test_path),
            validation_steps=valid_numb // BS,
            callbacks=callable,#class_weight=[0.01,0.99]
            max_queue_size=1)


if __name__ == '__main__':
    #train2(Augment=False) #交叉验证
    testpath='E:/11-ArMsMoTn/04-Dataset/03-2018Dataset/'
    train_totalyear(test_path=testpath,Augment=True)

