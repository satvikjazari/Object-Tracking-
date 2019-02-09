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
                eye_list = []
                eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
                for (x, y, w, h) in eyes:
                    img = cv2.rectangle(imagess, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    X = x + w
                    Y = y + h
                    eye_list.append(X + Y)
                if result.name is not 'Unknown':
                    if len(eye_list) > 1:
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
                # print(att)


                ### Attendance Code ###
                # try:
                #     if att.iloc[attendance_counter][result.name] is None:
                #         att.iloc[attendance_counter][result.name] = (str(datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute)) 
                # except:
                #     att.iloc[attendance_counter][result.name] = (str(datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute))
                # print('about to go in')
                # print(current_day - 1); print(att.columns.get_loc(result.name))
                # print(att.iloc[current_day - 1,att.columns.get_loc(result.name)])
                # print(type(att.iloc[current_day - 1,att.columns.get_loc(result.name)]))
                if att.iloc[current_day - 1,att.columns.get_loc(result.name)] == float(0) or att.iloc[current_day - 1,att.columns.get_loc(result.name)] == '0.0' :
                    # print('in the end game')
                    # print(current_day - 1); print(att.columns.get_loc(result.name))
                    att.iloc[current_day - 1,att.columns.get_loc(result.name)] = str(datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute)
                    att.to_csv(SAVE_PATH + current_month + '_entry.csv',index=False)









    # cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                # thickness=2, lineType=2)
    cv2.putText(frame, str(' ') , (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)
    # cv2.imwrite("face_(%d).jpg"% frame_count, frame)
    # df = pd.DataFrame(columns=['frame count','x1','y1','x2','y2'])

    # df = df.append([str(frame_count)])
    # df.to_csv('frame_file.csv')
    # return df_counter, df




                


def main(args):
    frame_count_orig=0
    frame_interval = 1  # Number of frames after which to run face detection
    fps_display_interval = 1  # seconds
    frame_rate = 1
    frame_count = 0
    playing=True
    # PATH = '/home/jazari/Aman_Workspace/facenet/face_yolo_video_test/'
    PATH = '/home/jazari/Documents/VID_20181016_143322.mp4'
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_EXPOSURE,-4)
    # video_capture.set(3,1280)
    # video_capture.set(4,1024)
    # video_capture.set(15, -8.0)
    # # video_capture.set(15, -6.0)
    # # video_capture.set(3,1280)
    # # video_capture.set(4, 1024)
    # # video_capture.set(15, -4.0)
    # face_recognition = face.Recognition()
    # start_time = time.time()
    # attendance_counter = 0


    
    


    # df = pd.DataFrame(columns=['Frame', 'x1', 'y1', 'x2', 'y2'])
    # df_counter = 0

    #My Code
    
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
    #end
    
    
    # files = os.listdir(PATH)
    # files.sort()
    
    '''arr = []
    for d, r, f in next(os.walk(PATH)):
    	for file in f:
    		arr.append(os.path.join(r, file))
    for f in arr:
    	print(files)'''



    #### CHANGES START FROM  HERE DIFFERENTIATING IT FROM real_time_face_recognition_facenet.py' #######

    



    # dirlist = glob.glob(PATH)
    # #print(dirlist)
    # content_list = []
    # for content in os.listdir(PATH):
    # 	content_list.append(content)
    # content_list.sort()
    # i = 0
    # b = PATH
    
    		

    
    # '''for listfile in files:
    # 	# print(listfile)
    # 	a = cv2.imread(listfile)
    # 	cv2.imshow(a)
    # 	cv2.waitKey(0)'''
    # # dirlist = glob.glob(PATH)
    # # pprint(dirlist)


    

    # if args.debug:
    #     print("Debug enabled")
    #     face.debug = True

    # for content in content_list:
    #     PATH = b
    #     PATH = PATH + str(content_list[i])
    #     # i = i + 1
    #     print(PATH)
    #     a = cv2.imread(PATH)
    #     #cv2.imshow('img', a)
    #     #cv2.waitKey(0)
    #     #cv2.destroyWindows('img')
    #     # print('1')
    #     i += 1
    #     # Capture frame-by-frame
    #     #ret, frame = video_capture.read()
    #     frame = a
    #     if (frame_count % frame_interval) == 0:
    #         faces = face_recognition.identify(frame)
    #         #faces = tfnet.return_predict(frame)
            
    #         # Check our current fps
    #         end_time = time.time()
    #         if (end_time - start_time) > fps_display_interval:
    #             frame_rate = int(frame_count / (end_time - start_time))
    #             start_time = time.time()

    #             frame_count = 0
    #             frame_count_orig += 1
    #     # print('2')
    #     df_counter, df = add_overlays(frame, faces, frame_rate, i, df, df_counter)

    #     frame_count += 1
    #     frame = cv2.resize(frame, (500, 400))
    #     # cv2.imshow('Video', frame)
    #     # cv2.waitKey(0)
    #     #time.sleep(15)

    #     # print('3')
    #     # print(i)
    #     #My code
    #     out = cv2.imwrite("face.jpg", frame)
    #     #out.write(frame)
    #     #time.sleep(25)
    #     playing=False
    #     #end

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    #     cv2.destroyAllWindows()

    # # When everything is done, release the capture
    # #video_capture.release()
    # df[['Frame', 'x1', 'y1', 'x2', 'y2']] = df[['Frame', 'x1', 'y1', 'x2', 'y2']].apply(pd.to_numeric)
    # df['xc'] = (df['x1'] + df['x2']) / 2
    # df['yc'] = (df['y1'] + df['y2']) / 2    
    # df.to_csv('facenet_test.csv')       




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

        ret, frame = video_capture.read()
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

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
 


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
