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
path='./dataset'

def augmento():
    dir=os.listdir(path)
    print(os.getcwd())
    for name in dir:
        #print(i)
        image_dir = os.listdir(path+'/'+str(name))
        # loc=os.listdir(path+'/'+str(name))
        # place = ''
        if image_dir.__len__()==1:
            # b='./'
            print(path+'/'+name+'/'+image_dir[0])
            save_path = './temporary'+'/'+name

            place=save_path
            original = cv2.imread(path+'/'+name+'/'+image_dir[0])
            img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

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
            plt.imsave(save_path+'/'+image_dir[0], img_new)
            counter += 1

            # Delete
            aug = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))
            img_new = aug.augment_image(img)
            plt.imsave(save_path + '/Delete.jpg', img_new)
            counter += 1


            # Rotate
            def rotateImage(angle):
                image_center = tuple(np.array(original.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
                result = cv2.warpAffine(original, rot_mat, original.shape[1::-1], flags=cv2.INTER_LINEAR)
                cv2.imwrite(save_path + '/' + str(angle) + 'Rotate.jpg', result)


            for i in range(-50, 50, 5):
                rotateImage(i)

            # Elastic Transformation
            aug = iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
            img_new = aug.augment_image(img)
            plt.imsave(save_path + '/Elastic Transformation.jpg', img_new)
            counter += 1
            print(place)
            print(path+'/'+str(name))
            if not os.path.exists('./Train'):
                os.makedirs('./Train')
            os.system('python align_dataset_mtcnn.py'+' '+'./temporary'+' '+'./Train' )
    name = str(datetime.datetime.now())
    os.system('python classifier.py TRAIN ./Train ./models/20180402-114759.pb ' + './new_pickle/'+'pickle'+'.pkl')


#print(l.__len__())
#augmento()