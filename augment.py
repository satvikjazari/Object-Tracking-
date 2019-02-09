from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
import os, shutil
import matplotlib.pyplot as plt
from skimage.transform import rescale
import warnings
from imgaug import augmenters as iaa
from scipy import misc
import sys
import argparse
import tensorflow as tf
import facenet
import detect_face
import random
import datetime
from time import sleep

warnings.filterwarnings("ignore")
from skimage import io


def augments(image , save_path):

    orignal = cv2.imread(image)
    img = cv2.cvtColor(orignal,cv2.COLOR_BGR2RGB)

    counter = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    def resize(val):

        small = cv2.resize(img, (0, 0), fx=val / 10, fy=val / 10)
        plt.imsave(save_path + '/Resize' + str(val) + '.jpg', small)

    for i in range(1, 10):
        resize(i)

    # Emboss
    aug = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))
    img_new = aug.augment_image(img)
    plt.imsave(save_path + '/Emboss.jpg', img_new)
    counter += 1

    # Gaussian Noise
    aug = iaa.GaussianBlur(sigma=(9.0))
    img_new = aug.augment_image(img)
    plt.imsave(save_path + '/Gaussian Noise.jpg', img_new)
    counter += 1

    # Average Blur
    aug = iaa.AverageBlur(k=(50, 50))
    img_new = aug.augment_image(img)
    plt.imsave(save_path + '/Average Blur.jpg', img_new)
    counter += 1

    # Median Blur
    # Odd Number below 16
    aug = iaa.MedianBlur(k=(15, 15))
    img_new = aug.augment_image(img)
    plt.imsave(save_path + '/Median Blur.jpg', img_new)
    counter += 1

    # Add Element Wise
    aug = iaa.AddElementwise((-40, 40))
    img_new = aug.augment_image(img)
    plt.imsave(save_path + '/Add Element Wise.jpg', img_new)
    counter += 1

    # Additive Gaussian Noise
    aug = iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))
    img_new = aug.augment_image(img)
    plt.imsave(save_path + '/Additive Gaussian Noise.jpg', img_new)
    counter += 1

    # Dropout
    aug = iaa.Dropout(p=(0, 0.2))
    img_new = aug.augment_image(img)
    plt.imsave(save_path + '/' + 'dp.jpg', img_new)
    counter += 1

    # Delete
    aug = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))
    img_new = aug.augment_image(img)
    plt.imsave(save_path + '/Delete.jpg', img_new)
    counter += 1

    # Rotate
    def rotateImage(angle):
        image_center = tuple(np.array(orignal.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(orignal, rot_mat, orignal.shape[1::-1], flags=cv2.INTER_LINEAR)
        cv2.imwrite(save_path + '/' + str(angle) + 'Rotate.jpg', result)

    for i in range(-50, 50, 5):
        rotateImage(i)

    # Elastic Transformation
    aug = iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
    img_new = aug.augment_image(img)
    plt.imsave(save_path + '/Elastic Transformation.jpg', img_new)
    counter += 1


#augments('./atest.jpg','./auto_try')