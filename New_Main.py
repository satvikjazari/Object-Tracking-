from network import network
import cv2
import tkinter as tk
import tkinter.messagebox
from PIL import Image, ImageTk
import threading
from queue import Queue
from file_auto import augmento

def update_win(im_label, q, size):
    while True:
        print('Displaying')
        image = q.get(block=True)
        image = cv2.resize(image, size)
        image1 = Image.fromarray(image)
        image2 = ImageTk.PhotoImage(image1)
        im_label.configure(image=image2)
        im_label.image = image2

def shutup(message):
    tkinter.messagebox.showinfo("Alert Message", message)


def main():
    win = tk.Tk()
    win.grid_propagate(False)
    win.rowconfigure(0, weight=1)
    win.columnconfigure(0, weight=1)
    mainframe = tk.Frame(master = win, bg = 'yellow')
    mainframe2 = tk.Frame(master=win, bg='orange')
    mainframe.grid(row=0, column=0, sticky='nsew')
    mainframe.rowconfigure(0, weight=1)
    mainframe2.grid(row=0, column=0, sticky='nsew')
    mainframe2.rowconfigure(0, weight=1)
    # win.rowconfigure(1, weight=1)
    mainframe.columnconfigure(0, weight=1)
    mainframe.columnconfigure(1, weight=1)
    mainframe2.columnconfigure(0, weight=1)
    mainframe2.columnconfigure(1, weight=1)
    # win.rowconfigure(2, weight=1)

    f1 = tk.Frame(master=mainframe)
    f2 = tk.Frame(master=mainframe)
    # f3 = tk.Frame(master=win)
    # f4 = tk.Frame(master=win)
    # bottom_panel = tk.Frame(master=win, bg='purple', height = "20")
    f1.grid(row = 0, column = 0, sticky = 'nsew')
    f2.grid(row = 0, column = 1, sticky = 'nsew')
    # f3.grid(row = 1, column = 0, sticky = 'nsew')
    # f4.grid(row = 1, column = 1, sticky = 'nsew')
    f1.pack_propagate(False)
    f2.pack_propagate(False)
    # f3.pack_propagate(False)
    # f4.pack_propagate(False)
    mainframe.tkraise()
    root_menu = tk.Menu(win)
    win.config(menu = root_menu)
    file_menu = tk.Menu(root_menu)
    root_menu.add_cascade(label = "File", menu = file_menu)
    file_menu.add_command(label = "Train", command = mainframe2.tkraise)






    im_label = tk.Label(master=f1, bg='green', width=500, height=500)
    im_label1 = tk.Label(master=f2, bg='blue', width=500, height=500)
    # im_label2 = tk.Label(master=f3, bg='blue', width=500, height=500)
    # im_label3 = tk.Label(master=f4, bg='blue', width=500, height=500)
    #
    im_label.pack(fill='both', expand=1)
    im_label1.pack(fill='both',expand=1)
    # im_label2.pack(fill='both',expand=1)
    # im_label3.pack(fill='both',expand=1)

    win.state('zoomed')
    win.update()
    w = f1.winfo_width()
    h = f1.winfo_height()
    image_panel_size = (w,h)
    fram_queue_1 = Queue(-1)
    # fram_queue_2 = Queue(-1)
    face_queue_1 = Queue(-1)
    # face_queue_2 = Queue(-1)

    # Camera 0
    t1 = threading.Thread(target=network, args=(fram_queue_1, 0, face_queue_1))
    t2 = threading.Thread(target=update_win, args=(im_label, fram_queue_1,image_panel_size))
    t3 = threading.Thread(target=update_win, args=(im_label1, face_queue_1,image_panel_size))
    # t1.start()
    t2.start()
    t3.start()

    # # Camera 1
    # t4 = threading.Thread(target=network, args=(fram_queue_2, 1, face_queue_2))
    # t5 = threading.Thread(target=update_win, args=(im_label1, fram_queue_2))
    # t6 = threading.Thread(target=update_win, args=(im_label3, face_queue_2))
    # t4.start()
    # t5.start()
    # t6.start()
    #
    #

    win.mainloop()


main()

