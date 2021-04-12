

import os
import cv2
import random
import numpy as np

from tqdm import tqdm

train_txt_path = './data/file_name.txt'  # 数据集images name索引txt
image_prefix = '/data/images'  # 图片


def cal_mean_std():
    '''统计图片的均值和方差'''
    global train_txt_path
    global image_prefix

    CNum = 10000 #取前100000张图片作为计算样本

    img_h, img_w = 32, 32
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, stdevs = [], []

    with open(train_txt_path, 'r') as f:
        lines = f.readlines() #读取全部image name
        random.shuffle(lines) # shuffle images

        for i in tqdm(range(CNum)): # 进度条
            file_name = lines[i].strip() + '.jpg'
            img_path = os.path.join(image_prefix, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_h, img_w)) # 将图片进行裁剪[32,32]
            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)

    imgs = imgs.astype(np.float32) / 255.

    for i in tqdm(range(3)):
        pixels = imgs[:,:,i,:].ravel() # flatten
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse() # BGR --> RGB
    stdevs.reverse()

    print("norm mean = {}".format(means))
    print("norm std = {}".format(stdevs))


def eigen_value_vector():
    '''统计图片数据集的特征值和特征向量'''
    global train_txt_path
    global image_prefix

    CNum = 10000
    img_h, img_w = 32, 32
    imgs = np.zeros([img_w, img_h, 3, 1])

    with open(train_txt_path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines) # shuffle images

        for i in tqdm(range(CNum)):
            file_name = lines[i].strip() + '.jpg'
            img_path = os.path.join(image_prefix, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_h, img_w))
            img = img[:,:,:, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)

    imgs = imgs.astype(np.float32) / 255.

    pixels = imgs[:, :, 2, :].ravel() # flatten
    scale_R = pixels - 0.42971808 # R
    pixels = imgs[:, :, 1, :].ravel() # flatten
    scale_G = pixels - 0.42560148 # G
    pixels = imgs[:, :, 0, :].ravel() # flatten
    scale_B = pixels - 0.4196568 # B

    cov = np.cov((scale_R, scale_G), scale_B) #求3个变量的协方差
    eig_val, eig_vec = np.linalg.eig(cov)
    print(eig_val)
    print(eig_vec)


