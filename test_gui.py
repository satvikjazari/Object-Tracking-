from tkinter import *
from tkinter import messagebox
from threading import Thread
import os
from New_Main import *
from New_Main import main as new_main
from file_auto import *
window = Tk()


window.title("SynergyLabs")

window.geometry('350x200')


def threaded_run():
    Thread(target=run).start()

def run():
    new_main()



def threaded_train():
    Thread(target=train).start()

def train():
    #import file_auto_original
    pass


# btn_run = Button(window, text='RUN', command=threaded_run)
# btn_train = Button(window, text='ADD', command=threaded_train)
# btn_run.grid(column=0, row=0)
# btn_train.grid(column=0, row=1)
# window.mainloop()
new_main()