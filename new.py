# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import sys
import time
import cv2
import matplotlib.pyplot as plt
import face
import pandas as pd
import os
import pathlib
import glob
from pprint import pprint
from darkflow.net.build import TFNet
import numpy as np
import datetime
import calendar
import pyttsx3
import win32com.client as wincl
from imutils.video import WebcamVideoStream, FPS
import imutils

option = {
    'model' : 'cfg/yolo-widerface.cfg',
    'load' : 'bin/yolo-widerface_final.weights',
    
    'gpu' : 0.0
}
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
tfnet = TFNet(option)
colors = [tuple(255 * np.random.rand(3)) for j in range(5)]


### Attendance Code ###

PATH_ATT = 'Dataset'
SAVE_PATH = 'entry/'


engine = pyttsx3.init()

counter = 1000


### ###





def create_csv(current_month_days):


    att = pd.DataFrame(None, columns=['Date'])

    for f in os.listdir(PATH_ATT):
        # print(f)
        att[f] = ""
        att= pd.DataFrame(data=0*np.ones((current_month_days,len(att.columns))),columns=list(att.columns))
        # df.to_csv(current_month + '.csv')
    return att



def add_overlays(frame, faces, frame_rate, attendance_counter, att, current_day, current_month):#, frame_count,df, df_counter):

    blur_threshold = 100
    
    
    

    
    if faces is not None:

        for result in faces:
            tl = (result.bounding_box[0], result.bounding_box[1])
            br = (result.bounding_box[2], result.bounding_box[3])

            # df_counter += 1
            # frame = cv2.rectangle(frame, tl, br, (1, 1, 1), 7)
            if result.name is not None:
                global counter
                file_name = './Custom_Entry/'
                 # + str(result.name) + str(counter) + '.jpg'
                imagess = frame[result.bounding_box[1]:result.bounding_box[3],
                          result.bounding_box[0]:result.bounding_box[2]]

                org = imagess.copy()
                gray = cv2.cvtColor(imagess, cv2.COLOR_BGR2GRAY)
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur < blur_threshold:
                    continue
                eye_list = []
                eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
                for (x, y, w, h) in eyes:
                    img = cv2.rectangle(imagess, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    X = x + w
                    Y = y + h
                    eye_list.append(X + Y)
                if result.name is not 'Unknown':
                    if len(eye_list) >= 1:
                        sv_pth = file_name+str(result.name) + '/'
                        if not os.path.exists(sv_pth):
                            os.makedirs(sv_pth)
                        pth = sv_pth+str(result.name)+str(datetime.datetime.now().hour) + '-' + str(datetime.datetime.now().minute)+'-'+str(datetime.datetime.now().second)+'.jpg'
                        print(pth)
                        cv2.imwrite(r''+pth, org)
                        counter += 1
                # engine.say('Welcome' + result.name)
                # engine.runAndWait()
                # speak = wincl.Dispatch("SAPI.SpVoice")
                # speak.Speak('Welcome' + result.name)
                # cv2.putText(frame, (result.name + str(result.conf)), (result.bounding_box[0], result.bounding_box[3]),
                            # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            # thickness=2, lineType=2)
                cv2.putText(frame, (result.name ), (result.bounding_box[0], result.bounding_box[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)            

                if att.iloc[current_day - 1,att.columns.get_loc(result.name)] == float(0) or att.iloc[current_day - 1,att.columns.get_loc(result.name)] == '0.0' :
                    # print('in the end game')
                    # print(current_day - 1); print(att.columns.get_loc(result.name))
                    att.iloc[current_day - 1,att.columns.get_loc(result.name)] = str(datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute)
                    att.to_csv(SAVE_PATH + current_month + '_entry.csv',index=False)

    cv2.putText(frame, str(frame_rate) , (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)



def main(args):
    frame_count_orig=0
    frame_interval = 1  # Number of frames after which to run face detection
    fps_display_interval = 1  # seconds
    frame_rate = 1
    frame_count = 0
    playing=True
    # PATH = '/home/jazari/Aman_Workspace/facenet/face_yolo_video_test/'
    PATH = 'C:/Users/jazari1/Desktop/Aman/facenet_attendance/face/sameer.mp4'
    # video_capture = cv2.VideoCapture(0)
    video_capture = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    # video_capture.set(15, -6.0)
    # video_capture.set(3,1280)
    # video_capture.set(4, 1024)
    # video_capture.set(15, -4.0)
    face_recognition = face.Recognition()
    start_time = time.time()
    attendance_counter = 0

    ###### CHANGES END. STARTING TO INTEGRATE VIDEO CODE ######

    if args.debug:
        print("Debug enabled")
        face.debug = True

    while True:
        # Capture frame-by-frame

        now = datetime.datetime.now()
        current_day = now.day
        # att = pd.read_csv('attendance.csv')
        
        current_month_days = calendar.monthrange(now.year, now.month)[1]
        current_month = now.strftime("%B")
        # print('inside loop')
        try:
            att = pd.read_csv(SAVE_PATH + current_month + '_entry.csv')
            # print('taking from try')
        except:    
            att = create_csv(current_month_days)
            # print('taking from catch')


        # print(att.iloc[current_day-1,  0])
        if int(att.iloc[current_day - 1,0]) == 0:
            att.iloc[current_day - 1,0] = now.day
            # print('Prining csv')
            att.to_csv(SAVE_PATH + current_month + '_entry.csv',index=False)
        # print(att)    

        # ret, frame = video_capture.read()
        frame = video_capture.read()
        # frame = imutils.resize(frame, width=400)
        # now = datetime.datetime.now()
        


        if now.strftime("%B") != current_month:
            current_month = now.strftime("%B")
            
            current_month_days = calendar.monthrange(now.year, now.month)[1]
            
            att = create_csv(current_month_days)

        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)

            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        
        ### Attendance Code ###

        if datetime.datetime.now().day != current_day:
            attendance_counter += 1


        add_overlays(frame, faces, frame_rate, attendance_counter, att, current_day, current_month)

        frame_count += 1
        # cv2.resizeWindow('image', 600,600)
        cv2.imshow('Video', frame)
        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fps.update()    

    # When everything is done, release the capture
    # video_capture.release()
    cv2.destroyAllWindows()
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

 


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
