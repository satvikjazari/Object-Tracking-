from network import network
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
from queue import Queue

# win = tk.Tk()
win = tk.Toplevel()
im_label = tk.Label(master=win, bg='green', width=500,height= 500)
# im_label1 = tk.Label(master=win, bg='blue', width=500,height= 500)
im_label2 = tk.Label(master=win, bg='blue', width=500,height= 500)
# im_label3 = tk.Label(master=win, bg='blue', width=500,height= 500)

im_label.grid(row=0, column = 0)
# im_label1.grid(row=1, column = 0)
im_label2.grid(row=0, column = 1)
# im_label3.grid(row=1, column = 1)

def update_win(im_label, q):
    while True:
        image = q.get(block=True)
        image = cv2.resize(image, (500, 500))
        image1 = Image.fromarray(image)
        image2 = ImageTk.PhotoImage(image1)
        im_label.configure(image=image2)
        im_label.image = image2

fram_queue_1 = Queue(-1)
fram_queue_2 = Queue(-1)
face_queue_1 = Queue(-1)
face_queue_2 = Queue(-1)



# Camera 0
t1 = threading.Thread(target=network, args=(fram_queue_1,0, face_queue_1))
t2 = threading.Thread(target=update_win, args=(im_label, fram_queue_1))
t3 = threading.Thread(target=update_win, args=(im_label2, face_queue_1))
t1.start()
t2.start()
t3.start()









# Camera 1
# t4 = threading.Thread(target=network, args=(fram_queue_2, 1, face_queue_2))
# t5 = threading.Thread(target=update_win, args=(im_label1, fram_queue_2))
# t6 = threading.Thread(target=update_win, args=(im_label3, face_queue_2))
# t4.start()
# t5.start()
# t6.start()
#



win.mainloop()