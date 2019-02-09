import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading


def update_win(cap, im_label):

    while True:
        ret, val = cap.read()
        if ret:
            print('True')
            image1 = Image.fromarray(val)
            image2 = ImageTk.PhotoImage(image1)
            im_label.configure(image=image2)
            im_label.image=image2


win = tk.Tk()

im_label2 = tk.Label(master=win, bg = 'green')
im_label = tk.Label(master=win, bg = 'green')

im_label.pack(expand=1, fill='both')
im_label2.pack(expand=1, fill='both')

win.geometry("600x600")

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture('VID_20181227_190138.mp4')

t1 = threading.Thread(target=update_win, args=(cap, im_label))
t1.start()

# t2 = threading.Thread(target=update_win, args=(cap2, im_label2))
# t2.start()

win.mainloop()

