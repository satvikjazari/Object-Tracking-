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

import numpy as np
import datetime
import calendar
from imutils.video import WebcamVideoStream, FPS
import imutils
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading

eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
# tfnet = TFNet(option)
colors = [tuple(255 * np.random.rand(3)) for j in range(5)]


### Attendance Code ###

PATH_ATT = 'Dataset'
SAVE_PATH = 'entry/'

win = tk.Tk()


im_label = tk.Label(master=win, bg = 'green')

im_label.pack(expand=1, fill='both')

win.geometry("600x600")

counter = 1000


### ###
emp = 0
def update_win(im_label, q):
    while True:
        image = q.get(block=True)
        image1 = Image.fromarray(image)
        image2 = ImageTk.PhotoImage(image1)
        im_label.configure(image=image2)
        im_label.image=image2



def create_csv(current_month_days):


    att = pd.DataFrame(None, columns=['Date'])

    for f in os.listdir(PATH_ATT):
        # print(f)
        att[f] = ""
        att= pd.DataFrame(data=0*np.ones((current_month_days,len(att.columns))), columns=list(att.columns))
        # df.to_csv(current_month + '.csv')
    return att



def add_overlays(frame, faces, frame_rate, attendance_counter, att, current_day, current_month):

    blur_threshold = 100
    
    
    

    
    if faces is not None:

        for result in faces:
            if result.name is not None:
                global counter
                file_name = './Custom_Entry/'

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

                        pth = sv_pth+str(result.name)+str(datetime.datetime.now().hour) + '-' + str(datetime.datetime.now().minute)+'-'+str(datetime.datetime.now().second)+'-'+str(result.conf)+'.jpg'

                        cv2.imwrite(r''+pth, org)
                        counter += 1
                cv2.putText(frame, (result.name ), (result.bounding_box[0], result.bounding_box[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)            

                if att.iloc[current_day - 1,att.columns.get_loc(result.name)] == float(0) or att.iloc[current_day - 1,att.columns.get_loc(result.name)] == '0.0' :

                    att.iloc[current_day - 1,att.columns.get_loc(result.name)] = str(datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute)
                    att.to_csv(SAVE_PATH + current_month + '_entry.csv',index=False)

    cv2.putText(frame, str(frame_rate) , (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)





                


def network( q):
    frame_count_orig=0
    frame_interval = 1  # Number of frames after which to run face detection
    fps_display_interval = 1  # seconds
    frame_rate = 1
    frame_count = 0
    playing=True
    # PATH = '/home/jazari/Aman_Workspace/facenet/face_yolo_video_test/'
    PATH = './sameer.mp4'
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
        emp=frame
        print('adding in queue')
        emp = cv2.cvtColor(emp, cv2.COLOR_BGR2RGB)
        # Adding a frame in Queue
        q.put(emp)
    

        # t1 = threading.Thread(target=update_win, args=(frame, im_label))
        # t1.start()
        # win.mainloop()

        





        # cv2.resizeWindow('image', 600,600)
        # cv2.imshow('Video', frame)
        # cv2.waitKey(0)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    #     fps.update()    

    # cv2.destroyAllWindows()
    # fps.stop()
    # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

 



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)






# cap = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture('VID_20181227_190138.mp4')



# t2 = threading.Thread(target=update_win, args=(cap2, im_label2))
# t2.start()
from queue import Queue

q = Queue(maxsize = 128)

t1 = threading.Thread(target=network, args=(q,))
t2 = threading.Thread(target = update_win, args = (im_label,q ))
t1.start()
t2.start()


win.mainloop()


