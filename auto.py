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
from tain_pkl import train
from augment import augments
warnings.filterwarnings("ignore")
from skimage import io
from align_dataset_mtcnn import align
def automated():
    Final_train = './Train'
    path = './dataset'
    aug_save = './temporary'
    pickle = './autopickle/'

    if not os.path.exists(aug_save):
        os.makedirs(aug_save)
    if not os.path.exists(pickle):
        os.makedirs(pickle)
    pname = os.listdir(path)
    dictionary = dict(zip(pname,np.arange(0,len(pname))))
    aname = os.listdir(aug_save)
    #AUGMENTATION CODE
    for i in dictionary:
        print(i)
        if i not in aname:
            print(path+'/'+os.listdir(path+'/'+i)[0])
            augments(path+'/'+i+'/'+os.listdir(path+'/'+i)[0],aug_save+'/'+i)
    #ALIGNMENT PART
    align(aug_save,Final_train)

    #FINAL TRAIN
    train(Final_train,pickle+'auto_final.pkl')

# automated()
