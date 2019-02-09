import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import pandas as pd
import os

option = {
    'model': 'cfg/yolo-widerface.cfg',
    'load': 'bin/yolo-widerface_final.weights',
    'threshold': 0.08,
    'gpu': 0.0
}

tfnet = TFNet(option)
# path = 'np.intersect1d(A, B)'
PATH = '/home/jazari/Aman Workspace/testing_frames/'
df = pd.DataFrame(columns=['Frame', 'x1', 'y1', 'x2', 'y2'])
df_counter = 0
#capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture(path)
content_list = []
for content in os.listdir(PATH):
    	content_list.append(content)
content_list.sort()
i = 0
b = PATH
frame_count = 1
    
colors = [tuple(255 * np.random.rand(3)) for j in range(5)]

for content in content_list:
	PATH = b
	PATH = PATH + str(content_list[i])
	stime = time.time()
	frame = cv2.imread(PATH)
	print(PATH)
    # ret, frame = capture.read()
	if True:
	    results = tfnet.return_predict(frame); print('hi')
	    for color, result in zip(colors, results):
	        tl = (result['topleft']['x'], result['topleft']['y'])
	        br = (result['bottomright']['x'], result['bottomright']['y'])
	        label = result['label']
	        df.loc[df_counter, ['Frame']] = frame_count
	        df.loc[df_counter, ['x1']] = result['topleft']['x']
	        df.loc[df_counter, ['y1']] = result['topleft']['y']
	        df.loc[df_counter, ['x2']] = result['bottomright']['x']
	        df.loc[df_counter, ['y2']] = result['bottomright']['y']
	        df_counter += 1; print('hi')
	        frame = cv2.rectangle(frame, tl, br, color, 7)
	        

			# cv2.FONT_HERSHEY_COMPLEX (image text height (float),image text color (r,g,b), image text thickness (int))
	        frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
	    cv2.imshow('frame', frame)
	    print('FPS {:.1f}'.format(1 / (time.time() - stime)))
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	else:
	    #capture.release()
	    cv2.destroyAllWindows()
	    break

	frame_count += 1; i+=1
df[['Frame', 'x1', 'y1', 'x2', 'y2']] = df[['Frame', 'x1', 'y1', 'x2', 'y2']].apply(pd.to_numeric)
df['xc'] = (df['x1'] + df['x2']) / 2
df['yc'] = (df['y1'] + df['y2']) / 2 
print("End ho gya")   
df.to_csv('facenet_test1.csv') 
print(pd.read_csv('facenet_test1.csv'))   

			
