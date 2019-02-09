from imutils.video import WebcamVideoStream, FPS
import face, calendar, datetime, time
from create_csv import create_csv
import pandas as pd
import imutils
import time, cv2
from Detect_Recognize import add_overlays
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
SAVE_PATH = 'entry/'
counter=1000
# NEW ARGUMENT ADDED AS kill_sig-----
def network(frame_q, source, face_q, kill_sig):
    print("network called")
    frame_count_orig = 0
    frame_interval = 1  # Number of frames after which to run face detection
    fps_display_interval = 1  # seconds
    frame_rate = 1
    frame_count = 0
    playing = True
    # PATH = '/home/jazari/Aman_Workspace/facenet/face_yolo_video_test/'
    # PATH = './sameer.mp4'
    video_capture = cv2.VideoCapture(source)
    # video_capture = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    face_recognition = face.Recognition()
    start_time = time.time()

    ###### CHANGES END. STARTING TO INTEGRATE VIDEO CODE ######

    while True:
        # Capture frame-by-frame
        if kill_sig.is_set():
            return
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

        ret, frame = video_capture.read()
        #___________________________________________FOR DEMO VIDEO

        #frame=imutils.rotate(frame,angle=270)
        #______________________________________________________
        # frame = video_capture.read()

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


        fc_q = add_overlays(frame, faces, frame_rate, att, current_day, current_month)



        frame_count += 1
        emp = frame
        # print('adding in queue')
        emp = cv2.cvtColor(emp, cv2.COLOR_BGR2RGB)
        # Adding a frame in Queue
        frame_q.put(emp)
        # print(frame_q.qsize())
        # print('yeah')
        if len(fc_q) > 10:

            face_q.put(cv2.cvtColor(fc_q, cv2.COLOR_BGR2RGB))

