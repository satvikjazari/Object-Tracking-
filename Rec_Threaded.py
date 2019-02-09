# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import cv2
from multiprocessing.pool import ThreadPool
from collections import deque
import argparse
import sys
import time
import cv2

import face
import pandas as pd
import os

import numpy as np
import datetime
import calendar
from imutils.video import FPS
from common import clock, draw_str, StatValue
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
PATH_ATT = 'Dataset'
SAVE_PATH = 'entry/'

counter = 1000


class DummyTask:
    def __init__(self, data):
        self.data = data
    def ready(self):
        return True
    def get(self):
        return self.data

if __name__ == '__main__':

    def create_csv(current_month_days):

        att = pd.DataFrame(None, columns=['Date'])

        for f in os.listdir(PATH_ATT):
            # print(f)
            att[f] = ""
            att = pd.DataFrame(data=0 * np.ones((current_month_days, len(att.columns))), columns=list(att.columns))
            # df.to_csv(current_month + '.csv')
        return att



    import sys

    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0
    cap = cv2.VideoCapture(0)
    frame_interval_detect = 1  # Number of frames after which to run face detection
    fps_display_interval = 1  # seconds
    frame_rate = 1
    frame_count = 0
    playing=True
    fps = FPS().start()
    face_recognition = face.Recognition()
    start_time = time.time()
    attendance_counter = 0


    def process_frame(frame, t0):
        faces = face_recognition.identify(frame)

        blur_threshold = 100

        if faces is not None:

            for result in faces:
                # tl = (result.bounding_box[0], result.bounding_box[1])
                # br = (result.bounding_box[2], result.bounding_box[3])

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
                            sv_pth = file_name + str(result.name) + '/'
                            if not os.path.exists(sv_pth):
                                os.makedirs(sv_pth)
                                # -----------------------------CHANGE HERE -------------ADDED CONFIDENCE--------------------
                            pth = sv_pth + str(result.name) + str(datetime.datetime.now().hour) + '-' + str(
                                datetime.datetime.now().minute) + '-' + str(datetime.datetime.now().second) + '-' + str(
                                result.conf) + '.jpg'
                            print(pth)
                            cv2.imwrite(r'' + pth, org)

                            counter += 1

                    if att.iloc[current_day - 1, att.columns.get_loc(result.name)] == float(0) or att.iloc[
                        current_day - 1, att.columns.get_loc(result.name)] == '0.0':
                        att.iloc[current_day - 1, att.columns.get_loc(result.name)] = str(
                            datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute)
                        att.to_csv(SAVE_PATH + current_month + '_entry.csv', index=False)

        cv2.putText(frame, str(frame_rate), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    thickness=2, lineType=2)
        return frame, t0


    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    pending = deque()

    threaded_mode = True

    latency = StatValue()
    frame_interval = StatValue()
    last_frame_time = clock()
    while True:

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
        if int(att.iloc[current_day - 1, 0]) == 0:
            att.iloc[current_day - 1, 0] = now.day
            # print('Prining csv')
            att.to_csv(SAVE_PATH + current_month + '_entry.csv', index=False)
        # print(att)

        # ret, frame = video_capture.read()
        # frame = video_capture.read()

        while len(pending) > 0 and pending[0].ready():
            res, t0 = pending.popleft().get()
            latency.update(clock() - t0)
            draw_str(res, (20, 20), "threaded      :  " + str(threaded_mode))
            draw_str(res, (20, 40), "latency        :  %.1f ms" % (latency.value*1000))
            draw_str(res, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value*1000))
            cv.imshow('threaded video', res)
        if len(pending) < threadn:
            ret, frame = cap.read()

            if now.strftime("%B") != current_month:
                current_month = now.strftime("%B")

                current_month_days = calendar.monthrange(now.year, now.month)[1]

                att = create_csv(current_month_days)
            #
            # if (frame_count % frame_interval_detect) == 0:
            #     faces = face_recognition.identify(frame)

                # Check our current fps
                end_time = time.time()
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = int(frame_count / (end_time - start_time))
                    start_time = time.time()
                    frame_count = 0

            ### Attendance Code ###

            if datetime.datetime.now().day != current_day:
                attendance_counter += 1

            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            if threaded_mode:
                task = pool.apply_async(process_frame, (frame.copy(), t))
            else:
                task = DummyTask(process_frame(frame, t))
                # task = DummyTask(process_frame(frame, t))

            pending.append(task)
        ch = cv.waitKey(1)
        if ch == ord(' '):
            threaded_mode = not threaded_mode
        if ch == 27:
            break
cv.destroyAllWindows()