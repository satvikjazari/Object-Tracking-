from network import network
import cv2
import tkinter as tk
import tkinter.messagebox
from PIL import Image, ImageTk
import threading
from queue import Queue
# from file_auto import augmento
from auto import automated
import imutils

def update_win(im_label, q, size, kill_sig):
    while True:
        if kill_sig.is_set():
            return
        image = q.get(block=True)
        image = cv2.resize(image, size)
        image1 = Image.fromarray(image)
        image2 = ImageTk.PhotoImage(image1)
        im_label.configure(image=image2)
        im_label.image = image2


class App:

    def __init__(self, root):
        self.win = root
        self.im_labels = []
        self.frames = []
        self.mainframe = None
        self.mainframe2 = None

        self.ui()

    def ui(self):
        self.win.grid_propagate(False)
        self.win.rowconfigure(0, weight=1)
        self.win.columnconfigure(0, weight=1)
        self.mainframe = tk.Frame(master=self.win, bg='yellow')
        self.mainframe2 = tk.Frame(master=self.win, bg='orange')
        self.mainframe.grid(row=0, column=0, sticky='nsew')
        self.mainframe.rowconfigure(0, weight=1)
        self.mainframe2.grid(row=0, column=0, sticky='nsew')
        self.mainframe2.rowconfigure(0, weight=1)
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.columnconfigure(1, weight=1)
        self.mainframe2.columnconfigure(0, weight=1)

        f1 = tk.Frame(master=self.mainframe)
        f2 = tk.Frame(master=self.mainframe)

        f1.grid(row=0, column=0, sticky='nsew')
        f2.grid(row=0, column=1, sticky='nsew')

        f1.pack_propagate(False)
        f2.pack_propagate(False)

        root_menu = tk.Menu(self.win)
        self.win.config(menu=root_menu)
        file_menu = tk.Menu(root_menu)
        root_menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Train", command=self.kill_app)

        self.im_label1 = tk.Label(master=f1, bg='green', width=500, height=500)
        self.im_label2 = tk.Label(master=f2, bg='blue', width=500, height=500)

        self.im_label1.pack(fill='both', expand=1)
        self.im_label2.pack(fill='both', expand=1)

        self.win.state('zoomed')
        self.win.update()
        w = f1.winfo_width()
        h = f1.winfo_height()
        self.image_panel_size = (w, h)

        self.mainframe2.tkraise()
        train_thread = threading.Thread(target=self.train_)
        self.b1 = tk.Button(self.mainframe2, text="Train", command=train_thread.start)
        self.b2 = tk.Button(self.mainframe2, text="Start", command=self.thread_scheduler)

        self.b1.pack(side='left')
        self.b2.pack(side='right')

    def train_(self):
        tkinter.messagebox.showinfo("Train", "Training started")

        automated()
        tkinter.messagebox.showinfo("Train", "Training Done")

    def thread_scheduler(self):
        self.mainframe.tkraise()
        fram_queue_1 = Queue(-1)
        # fram_queue_2 = Queue(-1)
        face_queue_1 = Queue(-1)
        self.kill_sig = threading.Event()
        t1 = threading.Thread(target=network, args=(fram_queue_1, 0, face_queue_1, self.kill_sig))
        t2 = threading.Thread(target=update_win, args=(self.im_label1, fram_queue_1, self.image_panel_size, self.kill_sig))
        t3 = threading.Thread(target=update_win, args=(self.im_label2, face_queue_1, self.image_panel_size,self.kill_sig))
        self.threads = [t1,t2,t3]
        t1.start()
        t2.start()
        t3.start()

    def kill_app(self):
        # TODO Close all the threads and check if they are alive, exit app.
        self.kill_sig.set()

        
        self.win.destroy()





win = tk.Tk()

x = App(win)

win.mainloop()
