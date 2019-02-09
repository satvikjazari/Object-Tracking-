import os
import cv2
import datetime

SAVE_PATH = 'entry/'
counter=1000
def add_overlays(frame, faces, frame_rate, att, current_day, current_month):
    blur_threshold = 100
    org = []
    eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
    #global counter
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

                if result.name is not 'Unknown' and len(eye_list) >= 1:
                    sv_pth = file_name + str(result.name) + '/'
                    if not os.path.exists(sv_pth):
                        os.makedirs(sv_pth)
                    pth = sv_pth + str(datetime.datetime.now().hour) + '-' + str(
                        datetime.datetime.now().minute) + '-' + str(datetime.datetime.now().second) + '-'
                    n_pth = str(result.conf) + '.jpg'
                    cv2.imwrite(r'' +pth+n_pth, org)
                    cv2.imwrite(r''+pth+n_pth,org)
                    cv2.imwrite(r''+pth+'_frame'+n_pth,frame)
                    counter += 1
                    cv2.putText(frame, (result.name), (result.bounding_box[0], result.bounding_box[3]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                thickness=2, lineType=2)
                if result.name is 'Unknown':
                    sv_pth = file_name + str(result.name) + '/'
                    if not os.path.exists(sv_pth):
                        os.makedirs(sv_pth)
                        # -----------------------------CHANGE HERE -------------ADDED CONFIDENCE--------------------
                    pth = sv_pth + str(datetime.datetime.now().hour) + '-' + str(datetime.datetime.now().minute) + '-' + str(datetime.datetime.now().second) + '-'
                    n_pth = str(result.conf) + '.jpg'
                    #print(pth)
                    cv2.imwrite(r'' + pth + n_pth, org)
                    cv2.imwrite(r'' + pth + '_frame' + n_pth, frame)
                    counter += 1
                    cv2.putText(frame, (result.name), (result.bounding_box[0], result.bounding_box[3]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                thickness=2, lineType=2)



                # cv2.putText(frame, (result.name), (result.bounding_box[0], result.bounding_box[3]),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                #             thickness=2, lineType=2)

                if att.iloc[current_day - 1, att.columns.get_loc(result.name)] == float(0) or att.iloc[
                    current_day - 1, att.columns.get_loc(result.name)] == '0.0':
                    att.iloc[current_day - 1, att.columns.get_loc(result.name)] = str(
                        datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute)
                    att.to_csv(SAVE_PATH + current_month + '_entry.csv', index=False)
    cv2.putText(frame, str(frame_rate), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)
    return org