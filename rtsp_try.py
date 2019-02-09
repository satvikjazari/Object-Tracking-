import cv2
import numpy as np
#from file_auto import *
import os
import datetime
#augment()
# frame = cv2.VideoCapture('rtsp://184.72.239.149/vod/mp4:BigBuckBunny_115k.mov')
#
# while frame.isOpened():
#     ret , vid = frame.read()
#     if ret:
#         cv2.imshow('win',vid)
#
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     else:
#         break
# frame.release()
# cv2.destroyAllWindows()
#s = [1,24,52,1,3]

#s = np.array([1,23,4,1,])
#os.system('python classifier.py TRAIN ./Train/ ./models/20180402-114759.pb'+' '+'./new_pickle/'+str(3)+'.pkl')
d=str(datetime.datetime.now())
print(d[:11])